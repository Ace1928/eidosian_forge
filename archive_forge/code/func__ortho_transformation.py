import numpy as np
from matplotlib import _api
def _ortho_transformation(zfront, zback):
    a = -(zfront + zback)
    b = -(zfront - zback)
    proj_matrix = np.array([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, -2, 0], [0, 0, a, b]])
    return proj_matrix