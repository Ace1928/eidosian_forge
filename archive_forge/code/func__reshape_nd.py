import numpy
import cupy
from cupy import _core
from cupy._core import internal
from cupyx.scipy.ndimage import _util
from cupyx.scipy import special
def _reshape_nd(arr, ndim, axis):
    """Promote a 1d array to ndim with non-singleton size along axis."""
    nd_shape = (1,) * axis + (arr.size,) + (1,) * (ndim - axis - 1)
    return arr.reshape(nd_shape)