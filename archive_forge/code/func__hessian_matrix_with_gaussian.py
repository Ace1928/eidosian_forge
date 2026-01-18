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
def _hessian_matrix_with_gaussian(image, sigma=1, mode='reflect', cval=0, order='rc'):
    """Compute the Hessian via convolutions with Gaussian derivatives.

    In 2D, the Hessian matrix is defined as:
        H = [Hrr Hrc]
            [Hrc Hcc]

    which is computed by convolving the image with the second derivatives
    of the Gaussian kernel in the respective r- and c-directions.

    The implementation here also supports n-dimensional data.

    Parameters
    ----------
    image : ndarray
        Input image.
    sigma : float or sequence of float, optional
        Standard deviation used for the Gaussian kernel, which sets the
        amount of smoothing in terms of pixel-distances. It is
        advised to not choose a sigma much less than 1.0, otherwise
        aliasing artifacts may occur.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    order : {'rc', 'xy'}, optional
        This parameter allows for the use of reverse or forward order of
        the image axes in gradient computation. 'rc' indicates the use of
        the first axis initially (Hrr, Hrc, Hcc), whilst 'xy' indicates the
        usage of the last axis initially (Hxx, Hxy, Hyy)

    Returns
    -------
    H_elems : list of ndarray
        Upper-diagonal elements of the hessian matrix for each pixel in the
        input image. In 2D, this will be a three element list containing [Hrr,
        Hrc, Hcc]. In nD, the list will contain ``(n**2 + n) / 2`` arrays.

    """
    image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    if image.ndim > 2 and order == 'xy':
        raise ValueError("order='xy' is only supported for 2D images.")
    if order not in ['rc', 'xy']:
        raise ValueError(f'unrecognized order: {order}')
    if np.isscalar(sigma):
        sigma = (sigma,) * image.ndim
    truncate = 8 if all((s > 1 for s in sigma)) else 100
    sq1_2 = 1 / math.sqrt(2)
    sigma_scaled = tuple((sq1_2 * s for s in sigma))
    common_kwargs = dict(sigma=sigma_scaled, mode=mode, cval=cval, truncate=truncate)
    gaussian_ = functools.partial(ndi.gaussian_filter, **common_kwargs)
    ndim = image.ndim
    orders = tuple(([0] * d + [1] + [0] * (ndim - d - 1) for d in range(ndim)))
    gradients = [gaussian_(image, order=orders[d]) for d in range(ndim)]
    axes = range(ndim)
    if order == 'xy':
        axes = reversed(axes)
    H_elems = [gaussian_(gradients[ax0], order=orders[ax1]) for ax0, ax1 in combinations_with_replacement(axes, 2)]
    return H_elems