import functools
from numpy.core import asarray, zeros, swapaxes, conjugate, take, sqrt
from . import _pocketfft_internal as pfi
from numpy.core.multiarray import normalize_axis_index
from numpy.core import overrides
def _cook_nd_args(a, s=None, axes=None, invreal=0):
    if s is None:
        shapeless = 1
        if axes is None:
            s = list(a.shape)
        else:
            s = take(a.shape, axes)
    else:
        shapeless = 0
    s = list(s)
    if axes is None:
        axes = list(range(-len(s), 0))
    if len(s) != len(axes):
        raise ValueError('Shape and axes have different lengths.')
    if invreal and shapeless:
        s[-1] = (a.shape[axes[-1]] - 1) * 2
    return (s, axes)