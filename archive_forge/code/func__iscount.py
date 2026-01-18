from statsmodels.compat.python import lzip
import numpy as np
from scipy.stats import norm
from statsmodels.tools.decorators import cache_readonly
def _iscount(X):
    """
    Given an array X, returns the column indices for count variables.

    Parameters
    ----------
    X : array_like
        A 1d or 2d array of numbers

    Examples
    --------
    >>> X = np.random.randint(0, 10, size=(15,5)).astype(float)
    >>> X[:,1:3] = np.random.randn(15,2)
    >>> ind = _iscount(X)
    >>> ind
    array([0, 3, 4])
    """
    X = np.asarray(X)
    remainder = np.logical_and(np.logical_and(np.all(X % 1.0 == 0, axis=0), X.var(0) != 0), np.all(X >= 0, axis=0))
    dummy = _isdummy(X)
    remainder = np.where(remainder)[0].tolist()
    for idx in dummy:
        remainder.remove(idx)
    return np.array(remainder)