from numbers import Number
import warnings
import numpy as np
import cupy
from cupy.cuda import cufft
from cupy.fft._fft import (_fft, _default_fft_func, hfft as _hfft,
def _assequence(x):
    """Convert scalars to a sequence, otherwise pass through ``x`` unchanged"""
    if isinstance(x, Number):
        return (x,)
    return x