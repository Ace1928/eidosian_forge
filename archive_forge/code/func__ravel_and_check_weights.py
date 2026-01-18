import operator
import warnings
import numpy
import cupy
from cupy import _core
from cupy._core import _accelerator
from cupy.cuda import cub
from cupy.cuda import common
from cupy.cuda import runtime
def _ravel_and_check_weights(a, weights):
    """ Check a and weights have matching shapes, and ravel both """
    if a.dtype == cupy.bool_:
        warnings.warn('Converting input from {} to {} for compatibility.'.format(a.dtype, cupy.uint8), RuntimeWarning, stacklevel=3)
        a = a.astype(cupy.uint8)
    if weights is not None:
        if not isinstance(weights, cupy.ndarray):
            raise ValueError('weights must be a cupy.ndarray')
        if weights.shape != a.shape:
            raise ValueError('weights should have the same shape as a.')
        weights = weights.ravel()
    a = a.ravel()
    return (a, weights)