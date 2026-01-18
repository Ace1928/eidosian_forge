from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass
import numpy as np
import pandas as pd
from patsy import dmatrix
from patsy.mgcv_cubic_splines import _get_all_sorted_knots
from statsmodels.tools.linalg import transf_constraints
class BSplines(AdditiveGamSmoother):
    """additive smooth components using B-Splines

    This creates and holds the B-Spline basis function for several
    components.

    Parameters
    ----------
    x : array_like, 1-D or 2-D
        underlying explanatory variable for smooth terms.
        If 2-dimensional, then observations should be in rows and
        explanatory variables in columns.
    df :  {int, array_like[int]}
        number of basis functions or degrees of freedom; should be equal
        in length to the number of columns of `x`; may be an integer if
        `x` has one column or is 1-D.
    degree : {int, array_like[int]}
        degree(s) of the spline; the same length and type rules apply as
        to `df`
    include_intercept : bool
        If False, then the basis functions are transformed so that they
        do not include a constant. This avoids perfect collinearity if
        a constant or several components are included in the model.
    constraints : {None, str, array}
        Constraints are used to transform the basis functions to satisfy
        those constraints.
        `constraints = 'center'` applies a linear transform to remove the
        constant and center the basis functions.
    variable_names : {list[str], None}
        The names for the underlying explanatory variables, x used in for
        creating the column and parameter names for the basis functions.
        If ``x`` is a pandas object, then the names will be taken from it.
    knot_kwds : None or list of dict
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


    Attributes
    ----------
    smoothers : list of univariate smooth component instances
    basis : design matrix, array of spline bases columns for all components
    penalty_matrices : list of penalty matrices, one for each smooth term
    dim_basis : number of columns in the basis
    k_variables : number of smooth components
    col_names : created names for the basis columns

    There are additional attributes about the specification of the splines
    and some attributes mainly for internal use.

    Notes
    -----
    A constant in the spline basis function can be removed in two different
    ways.
    The first is by dropping one basis column and normalizing the
    remaining columns. This is obtained by the default
    ``include_intercept=False, constraints=None``
    The second option is by using the centering transform which is a linear
    transformation of all basis functions. As a consequence of the
    transformation, the B-spline basis functions do not have locally bounded
    support anymore. This is obtained ``constraints='center'``. In this case
    ``include_intercept`` will be automatically set to True to avoid
    dropping an additional column.
    """

    def __init__(self, x, df, degree, include_intercept=False, constraints=None, variable_names=None, knot_kwds=None):
        if isinstance(degree, int):
            self.degrees = np.array([degree], dtype=int)
        else:
            self.degrees = degree
        if isinstance(df, int):
            self.dfs = np.array([df], dtype=int)
        else:
            self.dfs = df
        self.knot_kwds = knot_kwds
        self.constraints = constraints
        if constraints == 'center':
            include_intercept = True
        super().__init__(x, include_intercept=include_intercept, variable_names=variable_names)

    def _make_smoothers_list(self):
        smoothers = []
        for v in range(self.k_variables):
            kwds = self.knot_kwds[v] if self.knot_kwds else {}
            uv_smoother = UnivariateBSplines(self.x[:, v], df=self.dfs[v], degree=self.degrees[v], include_intercept=self.include_intercept[v], constraints=self.constraints, variable_name=self.variable_names[v], **kwds)
            smoothers.append(uv_smoother)
        return smoothers