import math
import numbers
import operator
import numpy
import cupy
from cupy import _core
from cupy.fft._fft import _cook_shape
from cupyx.scipy.fft import _fft
def _exp_factor_dct2(x, n, axis, norm_factor, n_truncate=None):
    """Twiddle & scaling factors for computation of DCT/DST-II via FFT."""
    if n_truncate is None:
        n_truncate = n
    tmp = cupy.empty((n_truncate,), dtype=x.dtype)
    _mult_factor_dct2(tmp.real, n, norm_factor, tmp)
    if x.ndim == 1:
        return tmp
    tmp_shape = [1] * x.ndim
    tmp_shape[axis] = n_truncate
    tmp_shape = tuple(tmp_shape)
    return tmp.reshape(tmp_shape)