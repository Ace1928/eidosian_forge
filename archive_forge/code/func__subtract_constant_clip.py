import numpy as np
from .._shared.utils import warn
from ..util import dtype_limits, invert, crop
from . import grayreconstruct, _util
from ._extrema_cy import _local_maxima
def _subtract_constant_clip(image, const_value):
    """Subtract constant from image while handling underflow issues."""
    min_dtype, max_dtype = dtype_limits(image, clip_negative=False)
    if const_value > max_dtype - min_dtype:
        raise ValueError('The subtracted constant is not compatiblewith the image data type.')
    result = image - const_value
    result[image < const_value + min_dtype] = min_dtype
    return result