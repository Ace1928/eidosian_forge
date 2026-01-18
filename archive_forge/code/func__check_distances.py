import math
import numbers
import os
import cupy
from ._util import _get_inttype
def _check_distances(distances, shape, dtype):
    if distances.shape != shape:
        raise RuntimeError('distances array has wrong shape')
    if distances.dtype != dtype:
        raise RuntimeError(f'distances array must have dtype: {dtype}')