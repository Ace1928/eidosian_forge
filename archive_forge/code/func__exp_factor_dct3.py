import math
import numbers
import operator
import numpy
import cupy
from cupy import _core
from cupy.fft._fft import _cook_shape
from cupyx.scipy.fft import _fft
def _exp_factor_dct3(x, n, axis, dtype, norm_factor):
    """Twiddle & scaling factors for computation of DCT/DST-III via FFT."""
    tmp = cupy.empty((n,), dtype=dtype)
    _mult_factor_dct3(tmp.real, n, norm_factor, tmp)
    if x.ndim == 1:
        return tmp
    tmp_shape = [1] * x.ndim
    tmp_shape[axis] = n
    tmp_shape = tuple(tmp_shape)
    return tmp.reshape(tmp_shape)