import numpy as np
from ..util.dtype import dtype_range, dtype_limits
from .._shared import utils
def _assert_non_negative(image):
    if np.any(image < 0):
        raise ValueError('Image Correction methods work correctly only on images with non-negative values. Use skimage.exposure.rescale_intensity.')