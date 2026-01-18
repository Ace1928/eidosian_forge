import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from .._shared.utils import _supported_float_type
def _cv_small_disk(image_size):
    """Generates a disk level set function.

    The disk covers half of the image along its smallest dimension.
    """
    res = np.ones(image_size)
    centerY = int((image_size[0] - 1) / 2)
    centerX = int((image_size[1] - 1) / 2)
    res[centerY, centerX] = 0.0
    radius = float(min(centerX, centerY)) / 2.0
    return (radius - distance(res)) / (radius * 3)