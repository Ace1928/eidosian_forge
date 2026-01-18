import numpy as np
from numpy.linalg import lstsq
def _lowess_mycube(t):
    """
    Fast matrix cube

    Parameters
    ----------
    t : ndarray
        Array that is cubed, elementwise and in-place

    Returns
    -------
    Nothing
    """
    t2 = t * t
    t *= t2