import numpy as np
from matplotlib import _api
def _proj_points(points, M):
    return np.column_stack(_proj_trans_points(points, M))