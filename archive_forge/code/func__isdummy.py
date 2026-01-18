from statsmodels.compat.python import lzip
import numpy as np
from scipy.stats import norm
from statsmodels.tools.decorators import cache_readonly
def _isdummy(X):
    """
    Given an array X, returns the column indices for the dummy variables.

    Parameters
    ----------
    X : array_like
        A 1d or 2d array of numbers

    Examples
    --------
    >>> X = np.random.randint(0, 2, size=(15,5)).astype(float)
    >>> X[:,1:3] = np.random.randn(15,2)
    >>> ind = _isdummy(X)
    >>> ind
    array([0, 3, 4])
    """
    X = np.asarray(X)
    if X.ndim > 1:
        ind = np.zeros(X.shape[1]).astype(bool)
    max = np.max(X, axis=0) == 1
    min = np.min(X, axis=0) == 0
    remainder = np.all(X % 1.0 == 0, axis=0)
    ind = min & max & remainder
    if X.ndim == 1:
        ind = np.asarray([ind])
    return np.where(ind)[0]