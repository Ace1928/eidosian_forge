import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from .._shared.utils import _supported_float_type
def _cv_energy(image, phi, mu, lambda1, lambda2):
    """Returns the total 'energy' of the current level set function.

    This corresponds to equation (7) of the paper by Pascal Getreuer,
    which is the weighted sum of the following:
    (A) the length of the contour produced by the zero values of the
    level set,
    (B) the area of the "foreground" (area of the image where the
    level set is positive),
    (C) the variance of the image inside the foreground,
    (D) the variance of the image outside of the foreground

    Each value is computed for each pixel, and then summed. The weight
    of (B) is set to 0 in this implementation.
    """
    H = _cv_heavyside(phi)
    avgenergy = _cv_difference_from_average_term(image, H, lambda1, lambda2)
    lenenergy = _cv_edge_length_term(phi, mu)
    return np.sum(avgenergy) + np.sum(lenenergy)