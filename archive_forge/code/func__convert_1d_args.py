import warnings
import numpy
import cupy
from cupy_backends.cuda.api import runtime
from cupy import _core
from cupy._core import internal
from cupyx.scipy.ndimage import _util
def _convert_1d_args(ndim, weights, origin, axis):
    if weights.ndim != 1 or weights.size < 1:
        raise RuntimeError('incorrect filter size')
    axis = internal._normalize_axis_index(axis, ndim)
    w_shape = [1] * ndim
    w_shape[axis] = weights.size
    weights = weights.reshape(w_shape)
    origins = [0] * ndim
    origins[axis] = _util._check_origin(origin, weights.size)
    return (weights, tuple(origins))