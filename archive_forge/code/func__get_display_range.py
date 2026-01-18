from collections import namedtuple
import numpy as np
from ...util import dtype as dtypes
from ...exposure import is_low_contrast
from ..._shared.utils import warn
from math import floor, ceil
def _get_display_range(image):
    """Return the display range for a given set of image properties.

    Parameters
    ----------
    image : array
        The input image.

    Returns
    -------
    lo, hi : same type as immin, immax
        The display range to be used for the input image.
    cmap : string
        The name of the colormap to use.
    """
    ip = _get_image_properties(image)
    immin, immax = (np.min(image), np.max(image))
    if ip.signed:
        magnitude = max(abs(immin), abs(immax))
        lo, hi = (-magnitude, magnitude)
        cmap = _diverging_colormap
    elif any(ip):
        _raise_warnings(ip)
        lo, hi = (immin, immax)
        cmap = _nonstandard_colormap
    else:
        lo = 0
        imtype = image.dtype.type
        hi = dtypes.dtype_range[imtype][1]
        cmap = _default_colormap
    return (lo, hi, cmap)