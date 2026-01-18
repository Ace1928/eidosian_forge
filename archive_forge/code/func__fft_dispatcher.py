import functools
from numpy.core import asarray, zeros, swapaxes, conjugate, take, sqrt
from . import _pocketfft_internal as pfi
from numpy.core.multiarray import normalize_axis_index
from numpy.core import overrides
def _fft_dispatcher(a, n=None, axis=None, norm=None):
    return (a,)