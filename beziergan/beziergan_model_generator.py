import numpy as np
from gan import GAN
import surf2stl
import yaml
import sys
import os


def rotateOp(theta):
    rad = np.radians(theta)
    return np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])

def scalingOp(s_x, s_y):
    return np.array([[s_x, 0.0], [0.0, s_y]])

def shiftOp(x, y):
    return np.array([x, y])

if len(sys.argv) < 2:
    print("Config file is not provided!")
    exit(-1)

with open(sys.argv[1]) as f:
    try:
        config = yaml.safe_load(os.path.expandvars(f.read()))
    except yaml.YAMLError as exc:
        print("Failed to load beziergan_model_generator_config.yaml")
        if hasattr(exc, 'problem_mark'):
            mark = exc.problem_mark
            print("Error position: ({}:{})".format(mark.line+1, mark.column+1))


rng = np.random.default_rng(config["rng-seed"])

# Model acquisition:
latent_dim = config["beziergan"]["model"]["latent-dims"]
noise_dim = config["beziergan"]["model"]["noise-dims"]
model_id = config["beziergan"]["model"]["id"]
bezier_degree = config["beziergan"]["model"]["bezier-degree"]
bounds = (0., 1.)
n_points = config["beziergan"]["model"]["n-points"] # Number of points in the airfoil

directory = '{}/{}_{}'.format(config["trained-models-folder"], latent_dim, noise_dim)
if model_id is not None:
    directory += '/{}'.format(model_id)

model = GAN(latent_dim, noise_dim, n_points, bezier_degree, bounds)
model.restore(directory=directory)
# end

airfoils_count = config["blade"]["airfoils-count"]

powerful_dims = np.array(config["beziergan"]["latent-transforms"]["powerful-dims"])
if config["beziergan"]["latent-transforms"].get("dim-to-vary", None) != None:
    dim_to_vary = config["beziergan"]["latent-transforms"]["dim-to-vary"] 
else:
    dim_to_vary = rng.choice(powerful_dims, 1)

latent = np.array([rng.random((latent_dim,))]*airfoils_count)
for i in range(airfoils_count):
    if config["beziergan"]["latent-transforms"]["invert-dim-to-vary"]:
        latent[i, dim_to_vary] = 1 - i / airfoils_count
    else:
        latent[i, dim_to_vary] = i / airfoils_count
    for dim, multiplier in config["beziergan"]["latent-transforms"]["dims-multipliers"].items():
        latent[i, int(dim)] = latent[i, int(dim)] * multiplier
    for dim, value in config["beziergan"]["latent-transforms"]["fixed-dims"].items():
        latent[i, int(dim)] = float(value)

X = model.synthesize(latent)

x_tip_scale = float(config["blade"]["affine-transforms"]["x-tip-scale"]) # rng.uniform(0.7, 0.9)
y_tip_scale = float(config["blade"]["affine-transforms"]["y-tip-scale"]) # rng.uniform(0.7, 0.9)
x_base_scale = float(config["blade"]["affine-transforms"]["x-base-scale"]) # rng.uniform(1.0, 1.1)
y_base_scale = float(config["blade"]["affine-transforms"]["y-base-scale"]) # rng.uniform(1.0, 1.5)
tip_twist = float(config["blade"]["affine-transforms"]["tip-twist"]) # rng.uniform(-15, 5)
angle_of_attack = float(config["blade"]["angle-of-attack"])
x_shift = float(config["blade"]["affine-transforms"]["x-base-shift"])
y_shift = float(config["blade"]["affine-transforms"]["y-base-shift"])
axis_x = float(config["blade"]["affine-transforms"]["x-tip-shift"])
axis_y = float(config["blade"]["affine-transforms"]["y-tip-shift"])

shifts_x = np.linspace(x_shift, axis_x + x_shift, airfoils_count)
shifts_y = np.linspace(y_shift, axis_y + y_shift, airfoils_count)
shiftOps = [shiftOp(x, y) for x, y in zip(shifts_x, shifts_y)]

scaleFactors_x = np.linspace(x_base_scale, x_tip_scale, airfoils_count)
scaleFactors_y = np.linspace(y_base_scale, y_tip_scale, airfoils_count)
scalingOps = [scalingOp(s_x, s_y) for s_x, s_y in zip(scaleFactors_x, scaleFactors_y)]
twistAngles = np.linspace(0 + angle_of_attack, tip_twist + angle_of_attack, airfoils_count)
twistOps = [rotateOp(theta) for theta in twistAngles]

for i in range(airfoils_count):
    X[i] = np.matmul(X[i], scalingOps[i])

for i in range(airfoils_count):
    X[i] = np.matmul(X[i], twistOps[i])

for i in range(airfoils_count):
    X[i] = X[i] + shiftOps[i]

x = np.vstack(X[:, :, 0])
#x = x * np.linspace(x_base_scale, x_tip_scale, airfoils_count)[:, np.newaxis]
y = np.vstack(X[:, :, 1])
#y = y * np.linspace(y_base_scale, y_tip_scale, airfoils_count)[:, np.newaxis]
z = np.array([[h] for h in np.linspace(0, float(config["blade"]["height"]), airfoils_count)])
matrix = np.matrix(np.zeros((airfoils_count, n_points)))
z = matrix + z

surf2stl.write(config["output-file"], x, y, z, mode='ascii')

