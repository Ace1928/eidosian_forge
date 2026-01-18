import math
import os
import cupy
import numpy as np
from ._util import _get_inttype
from ._pba_2d import (_check_distances, _check_indices,
def _generate_aniso_distance_computation():
    """
    Compute euclidean distance from current coordinate (ind_0, ind_1, ind_2) to
    the coordinates of the nearest point (z, y, x)."""
    return '\n    F tmp = static_cast<F>(z - ind_0) * sampling[0];\n    F sq_dist = tmp * tmp;\n    tmp = static_cast<F>(y - ind_1) * sampling[1];\n    sq_dist += tmp * tmp;\n    tmp = static_cast<F>(x - ind_2) * sampling[2];\n    sq_dist += tmp * tmp;\n    dist[i] = sqrt(static_cast<F>(sq_dist));\n    '