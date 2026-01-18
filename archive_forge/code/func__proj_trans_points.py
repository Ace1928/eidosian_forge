import numpy as np
from matplotlib import _api
def _proj_trans_points(points, M):
    xs, ys, zs = zip(*points)
    return proj_transform(xs, ys, zs, M)