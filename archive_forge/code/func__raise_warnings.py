from collections import namedtuple
import numpy as np
from ...util import dtype as dtypes
from ...exposure import is_low_contrast
from ..._shared.utils import warn
from math import floor, ceil
def _raise_warnings(image_properties):
    """Raise the appropriate warning for each nonstandard image type.

    Parameters
    ----------
    image_properties : ImageProperties named tuple
        The properties of the considered image.
    """
    ip = image_properties
    if ip.unsupported_dtype:
        warn('Non-standard image type; displaying image with stretched contrast.', stacklevel=3)
    if ip.low_data_range:
        warn('Low image data range; displaying image with stretched contrast.', stacklevel=3)
    if ip.out_of_range_float:
        warn('Float image out of standard range; displaying image with stretched contrast.', stacklevel=3)