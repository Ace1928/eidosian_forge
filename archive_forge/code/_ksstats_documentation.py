import numpy as np
import scipy.special
import scipy.special._ufuncs as scu
from scipy._lib._finite_differences import _derivative
Computes the PPF(or ISF) for the two-sided Kolmogorov-Smirnov distribution.

    Parameters
    ----------
    n : integer, array_like
        the number of samples
    q : float, array_like
        Probabilities, float between 0 and 1
    cdf : bool, optional
        whether to compute the PPF(default=true) or the ISF.

    Returns
    -------
    ppf : ndarray
        PPF (or ISF if cdf is False) at the specified locations

    The return value has shape the result of numpy broadcasting n and x.
    