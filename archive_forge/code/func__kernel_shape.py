import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import binary_erosion, convolve
from .._shared.utils import _supported_float_type, check_nD
from ..restoration.uft import laplacian
from ..util.dtype import img_as_float
def _kernel_shape(ndim, dim):
    """Return list of `ndim` 1s except at position `dim`, where value is -1.

    Parameters
    ----------
    ndim : int
        The number of dimensions of the kernel shape.
    dim : int
        The axis of the kernel to expand to shape -1.

    Returns
    -------
    shape : list of int
        The requested shape.

    Examples
    --------
    >>> _kernel_shape(2, 0)
    [-1, 1]
    >>> _kernel_shape(3, 1)
    [1, -1, 1]
    >>> _kernel_shape(4, -1)
    [1, 1, 1, -1]
    """
    shape = [1] * ndim
    shape[dim] = -1
    return shape