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
def _compute_derivatives(image, mode='constant', cval=0):
    """Compute derivatives in axis directions using the Sobel operator.

    Parameters
    ----------
    image : ndarray
        Input image.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.

    Returns
    -------
    derivatives : list of ndarray
        Derivatives in each axis direction.

    """
    derivatives = [ndi.sobel(image, axis=i, mode=mode, cval=cval) for i in range(image.ndim)]
    return derivatives