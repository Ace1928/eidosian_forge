import math
import numbers
import operator
import numpy
import cupy
from cupy import _core
from cupy.fft._fft import _cook_shape
from cupyx.scipy.fft import _fft
def _iterable_of_int(x, name=None):
    """Convert ``x`` to an iterable sequence of int."""
    if isinstance(x, numbers.Number):
        x = (x,)
    try:
        x = [operator.index(a) for a in x]
    except TypeError as e:
        name = name or 'value'
        raise ValueError(f'{name} must be a scalar or iterable of integers') from e
    return x