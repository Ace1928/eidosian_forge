import functools
import math
from itertools import combinations_with_replacement
import numpy as np
from scipy import ndimage as ndi
from scipy import spatial, stats
from .._shared.filters import gaussian
from .._shared.utils import _supported_float_type, safe_as_int, warn
from ..transform import integral_image
from ..util import img_as_float
from ._hessian_det_appx import _hessian_matrix_det
from .corner_cy import _corner_fast, _corner_moravec, _corner_orientations
from .peak import peak_local_max
from .util import _prepare_grayscale_input_2D, _prepare_grayscale_input_nD
def _symmetric_image(S_elems):
    """Convert the upper-diagonal elements of a matrix to the full
    symmetric matrix.

    Parameters
    ----------
    S_elems : list of array
        The upper-diagonal elements of the matrix, as returned by
        `hessian_matrix` or `structure_tensor`.

    Returns
    -------
    image : array
        An array of shape ``(M, N[, ...], image.ndim, image.ndim)``,
        containing the matrix corresponding to each coordinate.
    """
    image = S_elems[0]
    symmetric_image = np.zeros(image.shape + (image.ndim, image.ndim), dtype=S_elems[0].dtype)
    for idx, (row, col) in enumerate(combinations_with_replacement(range(image.ndim), 2)):
        symmetric_image[..., row, col] = S_elems[idx]
        symmetric_image[..., col, row] = S_elems[idx]
    return symmetric_image