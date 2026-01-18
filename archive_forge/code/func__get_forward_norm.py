import functools
from numpy.core import asarray, zeros, swapaxes, conjugate, take, sqrt
from . import _pocketfft_internal as pfi
from numpy.core.multiarray import normalize_axis_index
from numpy.core import overrides
def _get_forward_norm(n, norm):
    if n < 1:
        raise ValueError(f'Invalid number of FFT data points ({n}) specified.')
    if norm is None or norm == 'backward':
        return 1
    elif norm == 'ortho':
        return sqrt(n)
    elif norm == 'forward':
        return n
    raise ValueError(f'Invalid norm value {norm}; should be "backward","ortho" or "forward".')