import math
import os
import cupy
import numpy as np
from ._util import _get_inttype
from ._pba_2d import (_check_distances, _check_indices,
Fused decode3d and distance computation.

    This kernel is for use when `return_distances=True`, but
    `return_indices=False`. It replaces the separate calls to
    `_get_decode3d_kernel` and `_get_distance_kernel`, avoiding the overhead of
    generating full arrays containing the coordinates since the coordinate
    arrays are not going to be returned.
    