import math
import numpy as np
from .._shared.filters import gaussian
from .._shared.utils import convert_to_float
from ._warps import resize
def _smooth(image, sigma, mode, cval, channel_axis):
    """Return image with each channel smoothed by the Gaussian filter."""
    smoothed = np.empty_like(image)
    if channel_axis is not None:
        channel_axis = channel_axis % image.ndim
        sigma = (sigma,) * (image.ndim - 1)
    else:
        channel_axis = None
    gaussian(image, sigma=sigma, out=smoothed, mode=mode, cval=cval, channel_axis=channel_axis)
    return smoothed