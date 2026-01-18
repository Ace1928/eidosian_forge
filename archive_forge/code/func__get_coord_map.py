import numpy
import cupy
import cupy._core.internal
from cupyx.scipy.ndimage import _spline_prefilter_core
from cupyx.scipy.ndimage import _spline_kernel_weights
from cupyx.scipy.ndimage import _util
def _get_coord_map(ndim, nprepad=0):
    """Extract target coordinate from coords array (for map_coordinates).

    Notes
    -----
    Assumes the following variables have been initialized on the device::

        coords (ndarray): array of shape (ncoords, ndim) containing the target
            coordinates.
        c_j: variables to hold the target coordinates

    computes::

        c_j = coords[i + j * ncoords];

    ncoords is determined by the size of the output array, y.
    y will be indexed by the CIndexer, _ind.
    Thus ncoords = _ind.size();

    """
    ops = []
    ops.append('ptrdiff_t ncoords = _ind.size();')
    pre = f' + (W){nprepad}' if nprepad > 0 else ''
    for j in range(ndim):
        ops.append(f'\n    W c_{j} = coords[i + {j} * ncoords]{pre};')
    return ops