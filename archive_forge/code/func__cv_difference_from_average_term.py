import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from .._shared.utils import _supported_float_type
def _cv_difference_from_average_term(image, Hphi, lambda_pos, lambda_neg):
    """Returns the 'energy' contribution due to the difference from
    the average value within a region at each point.
    """
    c1, c2 = _cv_calculate_averages(image, Hphi)
    Hinv = 1.0 - Hphi
    return lambda_pos * (image - c1) ** 2 * Hphi + lambda_neg * (image - c2) ** 2 * Hinv