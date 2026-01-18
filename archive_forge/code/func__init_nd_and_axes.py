import math
import cupy
from cupy._core import internal
from cupyx.scipy import fft
from cupyx.scipy.ndimage import _filters
from cupyx.scipy.ndimage import _util
def _init_nd_and_axes(x, axes):
    axes = internal._normalize_axis_indices(axes, x.ndim, sort_axes=False)
    if not len(axes):
        raise ValueError('when provided, axes cannot be empty')
    if any((x.shape[ax] < 1 for ax in axes)):
        raise ValueError('invalid number of data points specified')
    return axes