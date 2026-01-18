from __future__ import annotations
from statsmodels.compat.python import lzip
from functools import reduce
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.base.data import handle_data
from statsmodels.base.optimizer import Optimizer
import statsmodels.base.wrapper as wrap
from statsmodels.formula import handle_formula_data
from statsmodels.stats.contrast import (
from statsmodels.tools.data import _is_using_pandas
from statsmodels.tools.decorators import (
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import nan_dot, recipr
from statsmodels.tools.validation import bool_like
def _get_wald_nonlinear(self, func, deriv=None):
    """Experimental method for nonlinear prediction and tests

        Parameters
        ----------
        func : callable, f(params)
            nonlinear function of the estimation parameters. The return of
            the function can be vector valued, i.e. a 1-D array
        deriv : function or None
            first derivative or Jacobian of func. If deriv is None, then a
            numerical derivative will be used. If func returns a 1-D array,
            then the `deriv` should have rows corresponding to the elements
            of the return of func.

        Returns
        -------
        nl : instance of `NonlinearDeltaCov` with attributes and methods to
            calculate the results for the prediction or tests

        """
    from statsmodels.stats._delta_method import NonlinearDeltaCov
    func_args = None
    nl = NonlinearDeltaCov(func, self.params, self.cov_params(), deriv=deriv, func_args=func_args)
    return nl