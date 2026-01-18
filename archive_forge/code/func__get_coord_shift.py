import numpy
import cupy
import cupy._core.internal
from cupyx.scipy.ndimage import _spline_prefilter_core
from cupyx.scipy.ndimage import _spline_kernel_weights
from cupyx.scipy.ndimage import _util
def _get_coord_shift(ndim, nprepad=0):
    """Compute target coordinate based on a shift.

    Notes
    -----
    Assumes the following variables have been initialized on the device::

        in_coord[ndim]: array containing the source coordinate
        shift[ndim]: array containing the zoom for each axis

    computes::

        c_j = in_coord[j] - shift[j]

    """
    ops = []
    pre = f' + (W){nprepad}' if nprepad > 0 else ''
    for j in range(ndim):
        ops.append(f'\n    W c_{j} = (W)in_coord[{j}] - shift[{j}]{pre};')
    return ops