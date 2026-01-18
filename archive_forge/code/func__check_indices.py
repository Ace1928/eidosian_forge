import math
import numbers
import os
import cupy
from ._util import _get_inttype
def _check_indices(indices, shape, itemsize):
    if indices.shape != shape:
        raise RuntimeError('indices array has wrong shape')
    if indices.dtype.kind not in 'iu':
        raise RuntimeError('indices array must have an integer dtype')
    elif indices.dtype.itemsize < itemsize:
        raise RuntimeError(f'indices dtype must have itemsize > {itemsize}')