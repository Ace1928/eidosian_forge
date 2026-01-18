from numbers import Number
import warnings
import numpy as np
import cupy
from cupy.cuda import cufft
from cupy.fft._fft import (_fft, _default_fft_func, hfft as _hfft,
def _implements(scipy_func):
    """Decorator adds function to the dictionary of implemented functions"""

    def inner(func):
        _implemented[scipy_func] = func
        return func
    return inner