import math
import numbers
import operator
import numpy
import cupy
from cupy import _core
from cupy.fft._fft import _cook_shape
from cupyx.scipy.fft import _fft
def _reshuffle_dct3(y, n, axis, dst):
    """Reorder entries to allow computation of DCT/DST-II via FFT."""
    x = cupy.empty_like(y)
    n_half = (n + 1) // 2
    sl_even = [slice(None)] * y.ndim
    sl_even[axis] = slice(0, None, 2)
    sl_even = tuple(sl_even)
    sl_half = [slice(None)] * y.ndim
    sl_half[axis] = slice(0, n_half)
    x[sl_even] = y[tuple(sl_half)]
    sl_odd = [slice(None)] * y.ndim
    sl_odd[axis] = slice(1, None, 2)
    sl_odd = tuple(sl_odd)
    sl_half[axis] = slice(-1, n_half - 1, -1)
    if dst:
        x[sl_odd] = -y[tuple(sl_half)]
    else:
        x[sl_odd] = y[tuple(sl_half)]
    return x