from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass
import numpy as np
import pandas as pd
from patsy import dmatrix
from patsy.mgcv_cubic_splines import _get_all_sorted_knots
from statsmodels.tools.linalg import transf_constraints
class UnivariateBSplines(UnivariateGamSmoother):
    """B-Spline single smooth component

    This creates and holds the B-Spline basis function for one
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
    variable_name : {None, str}
        The name for the underlying explanatory variable, x, used in for
        creating the column and parameter names for the basis functions.
    covder2_kwds : {None, dict}
        options for computing the penalty matrix from the second derivative
        of the spline.
    knot_kwds : {None, list[dict]}
        option for the knot selection.
        By default knots are selected in the same way as in patsy, however the
        number of knots is independent of keeping or removing the constant.
        Interior knot selection is based on quantiles of the data and is the
        same in patsy and mgcv. Boundary points are at the limits of the data
        range.
        The available options use with `get_knots_bsplines` are

        - knots : None or array
          interior knots
        - spacing : 'quantile' or 'equal'
        - lower_bound : None or float
          location of lower boundary knots, all boundary knots are at the same
          point
        - upper_bound : None or float
          location of upper boundary knots, all boundary knots are at the same
          point
        - all_knots : None or array
          If all knots are provided, then those will be taken as given and
          all other options will be ignored.
    """

    def __init__(self, x, df, degree=3, include_intercept=False, constraints=None, variable_name='x', covder2_kwds=None, **knot_kwds):
        self.degree = degree
        self.df = df
        self.include_intercept = include_intercept
        self.knots = get_knots_bsplines(x, degree=degree, df=df, **knot_kwds)
        self.covder2_kwds = covder2_kwds if covder2_kwds is not None else {}
        super().__init__(x, constraints=constraints, variable_name=variable_name)

    def _smooth_basis_for_single_variable(self):
        basis, der_basis, der2_basis = _eval_bspline_basis(self.x, self.knots, self.degree, include_intercept=self.include_intercept)
        cov_der2 = get_covder2(self, skip_ctransf=True, **self.covder2_kwds)
        return (basis, der_basis, der2_basis, cov_der2)

    def transform(self, x_new, deriv=0, skip_ctransf=False):
        """create the spline basis for new observations

        The main use of this stateful transformation is for prediction
        using the same specification of the spline basis.

        Parameters
        ----------
        x_new : ndarray
            observations of the underlying explanatory variable
        deriv : int
            which derivative of the spline basis to compute
            This is an options for internal computation.
        skip_ctransf : bool
            whether to skip the constraint transform
            This is an options for internal computation.

        Returns
        -------
        basis : ndarray
            design matrix for the spline basis for given ``x_new``
        """
        if x_new is None:
            x_new = self.x
        exog = _eval_bspline_basis(x_new, self.knots, self.degree, deriv=deriv, include_intercept=self.include_intercept)
        ctransf = getattr(self, 'ctransf', None)
        if ctransf is not None and (not skip_ctransf):
            exog = exog.dot(self.ctransf)
        return exog