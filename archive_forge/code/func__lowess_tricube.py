import numpy as np
from numpy.linalg import lstsq
def _lowess_tricube(t):
    """
    The _tricube function applied to a numpy array.
    The tricube function is (1-abs(t)**3)**3.

    Parameters
    ----------
    t : ndarray
        Array the tricube function is applied to elementwise and
        in-place.

    Returns
    -------
    Nothing
    """
    t[:] = np.absolute(t)
    _lowess_mycube(t)
    t[:] = np.negative(t)
    t += 1
    _lowess_mycube(t)