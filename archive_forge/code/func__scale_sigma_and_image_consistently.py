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
def _scale_sigma_and_image_consistently(image, sigma, multichannel, rescale_sigma):
    """If the ``image`` is rescaled, also rescale ``sigma`` consistently.

    Images that are not floating point will be rescaled via ``img_as_float``.
    Half-precision images will be promoted to single precision.
    """
    if multichannel:
        if isinstance(sigma, numbers.Number) or sigma is None:
            sigma = [sigma] * image.shape[-1]
        elif len(sigma) != image.shape[-1]:
            raise ValueError('When channel_axis is not None, sigma must be a scalar or have length equal to the number of channels')
    if image.dtype.kind != 'f':
        if rescale_sigma:
            range_pre = image.max() - image.min()
        image = img_as_float(image)
        if rescale_sigma:
            range_post = image.max() - image.min()
            scale_factor = range_post / range_pre
            if multichannel:
                sigma = [s * scale_factor if s is not None else s for s in sigma]
            elif sigma is not None:
                sigma *= scale_factor
    elif image.dtype == np.float16:
        image = image.astype(np.float32)
    return (image, sigma)