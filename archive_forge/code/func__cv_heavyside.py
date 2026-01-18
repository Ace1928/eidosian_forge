import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from .._shared.utils import _supported_float_type
def _cv_heavyside(x, eps=1.0):
    """Returns the result of a regularised heavyside function of the
    input value(s).
    """
    return 0.5 * (1.0 + 2.0 / np.pi * np.arctan(x / eps))