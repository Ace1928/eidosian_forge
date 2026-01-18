from functools import partial
import numpy as np
from scipy import stats
from statsmodels.tools.validation import string_like
from ._lilliefors_critical_values import (critical_values,
from .tabledist import TableDist
def _make_asymptotic_function(params):
    """
    Generates an asymptotic distribution callable from a param matrix

    Polynomial is a[0] * x**(-1/2) + a[1] * x**(-1) + a[2] * x**(-3/2)

    Parameters
    ----------
    params : ndarray
        Array with shape (nalpha, 3) where nalpha is the number of
        significance levels
    """

    def f(n):
        poly = np.array([1, np.log(n), np.log(n) ** 2])
        return np.exp(poly.dot(params.T))
    return f