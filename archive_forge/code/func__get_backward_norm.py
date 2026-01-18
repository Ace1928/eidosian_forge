import functools
from numpy.core import asarray, zeros, swapaxes, conjugate, take, sqrt
from . import _pocketfft_internal as pfi
from numpy.core.multiarray import normalize_axis_index
from numpy.core import overrides
def _get_backward_norm(n, norm):
    if n < 1:
        raise ValueError(f'Invalid number of FFT data points ({n}) specified.')
    if norm is None or norm == 'backward':
        return n
    elif norm == 'ortho':
        return sqrt(n)
    elif norm == 'forward':
        return 1
    raise ValueError(f'Invalid norm value {norm}; should be "backward", "ortho" or "forward".')