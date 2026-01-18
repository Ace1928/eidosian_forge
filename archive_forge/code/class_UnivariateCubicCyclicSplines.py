from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass
import numpy as np
import pandas as pd
from patsy import dmatrix
from patsy.mgcv_cubic_splines import _get_all_sorted_knots
from statsmodels.tools.linalg import transf_constraints
class UnivariateCubicCyclicSplines(UnivariateGamSmoother):
    """cyclic cubic regression spline single smooth component

    This creates and holds the Cyclic CubicSpline basis function for one
    component.

    Parameters
    ----------
    x : ndarray, 1-D
        underlying explanatory variable for smooth terms.
    df : int
        number of basis functions or degrees of freedom
    degree : int
        degree of the spline
    include_intercept : bool
        If False, then the basis functions are transformed so that they
        do not include a constant. This avoids perfect collinearity if
        a constant or several components are included in the model.
    constraints : {None, str, array}
        Constraints are used to transform the basis functions to satisfy
        those constraints.
        `constraints = 'center'` applies a linear transform to remove the
        constant and center the basis functions.
    variable_name : None or str
        The name for the underlying explanatory variable, x, used in for
        creating the column and parameter names for the basis functions.
    """

    def __init__(self, x, df, constraints=None, variable_name='x'):
        self.degree = 3
        self.df = df
        self.x = x
        self.knots = _equally_spaced_knots(x, df)
        super().__init__(x, constraints=constraints, variable_name=variable_name)

    def _smooth_basis_for_single_variable(self):
        basis = dmatrix('cc(x, df=' + str(self.df) + ') - 1', {'x': self.x})
        self.design_info = basis.design_info
        n_inner_knots = self.df - 2 + 1
        all_knots = _get_all_sorted_knots(self.x, n_inner_knots=n_inner_knots, inner_knots=None, lower_bound=None, upper_bound=None)
        b, d = self._get_b_and_d(all_knots)
        s = self._get_s(b, d)
        return (basis, None, None, s)

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

    def _get_s(self, b, d):
        return d.T.dot(np.linalg.inv(b)).dot(d)

    def transform(self, x_new):
        exog = dmatrix(self.design_info, {'x': x_new})
        if self.ctransf is not None:
            exog = exog.dot(self.ctransf)
        return exog