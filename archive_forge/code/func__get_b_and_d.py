from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass
import numpy as np
import pandas as pd
from patsy import dmatrix
from patsy.mgcv_cubic_splines import _get_all_sorted_knots
from statsmodels.tools.linalg import transf_constraints
def _get_b_and_d(self, knots):
    """Returns mapping of cyclic cubic spline values to 2nd derivatives.

        .. note:: See 'Generalized Additive Models', Simon N. Wood, 2006,
           pp 146-147

        Parameters
        ----------
        knots : ndarray
            The 1-d array knots used for cubic spline parametrization,
            must be sorted in ascending order.

        Returns
        -------
        b : ndarray
            Array for mapping cyclic cubic spline values at knots to
            second derivatives.
        d : ndarray
            Array for mapping cyclic cubic spline values at knots to
            second derivatives.

        Notes
        -----
        The penalty matrix is equal to ``s = d.T.dot(b^-1).dot(d)``
        """
    h = knots[1:] - knots[:-1]
    n = knots.size - 1
    b = np.zeros((n, n))
    d = np.zeros((n, n))
    b[0, 0] = (h[n - 1] + h[0]) / 3.0
    b[0, n - 1] = h[n - 1] / 6.0
    b[n - 1, 0] = h[n - 1] / 6.0
    d[0, 0] = -1.0 / h[0] - 1.0 / h[n - 1]
    d[0, n - 1] = 1.0 / h[n - 1]
    d[n - 1, 0] = 1.0 / h[n - 1]
    for i in range(1, n):
        b[i, i] = (h[i - 1] + h[i]) / 3.0
        b[i, i - 1] = h[i - 1] / 6.0
        b[i - 1, i] = h[i - 1] / 6.0
        d[i, i] = -1.0 / h[i - 1] - 1.0 / h[i]
        d[i, i - 1] = 1.0 / h[i - 1]
        d[i - 1, i] = 1.0 / h[i - 1]
    return (b, d)