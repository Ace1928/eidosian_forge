import functools
from math import ceil
import numbers
import scipy.stats
import numpy as np
from ..util.dtype import img_as_float
from .._shared import utils
from .._shared.utils import _supported_float_type, warn
from ._denoise_cy import _denoise_bilateral, _denoise_tv_bregman
from .. import color
from ..color.colorconv import ycbcr_from_rgb
def _rescale_sigma_rgb2ycbcr(sigmas):
    """Convert user-provided noise standard deviations to YCbCr space.

    Notes
    -----
    If R, G, B are linearly independent random variables and a1, a2, a3 are
    scalars, then random variable C:
        C = a1 * R + a2 * G + a3 * B
    has variance, var_C, given by:
        var_C = a1**2 * var_R + a2**2 * var_G + a3**2 * var_B
    """
    if sigmas[0] is None:
        return sigmas
    sigmas = np.asarray(sigmas)
    rgv_variances = sigmas * sigmas
    for i in range(3):
        scalars = ycbcr_from_rgb[i, :]
        var_channel = np.sum(scalars * scalars * rgv_variances)
        sigmas[i] = np.sqrt(var_channel)
    return sigmas