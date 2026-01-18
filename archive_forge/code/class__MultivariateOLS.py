import numpy as np
from numpy.linalg import eigvals, inv, solve, matrix_rank, pinv, svd
from scipy import stats
import pandas as pd
from patsy import DesignInfo
from statsmodels.compat.pandas import Substitution
from statsmodels.base.model import Model
from statsmodels.iolib import summary2
class _MultivariateOLS(Model):
    """
    Multivariate linear model via least squares


    Parameters
    ----------
    endog : array_like
        Dependent variables. A nobs x k_endog array where nobs is
        the number of observations and k_endog is the number of dependent
        variables
    exog : array_like
        Independent variables. A nobs x k_exog array where nobs is the
        number of observations and k_exog is the number of independent
        variables. An intercept is not included by default and should be added
        by the user (models specified using a formula include an intercept by
        default)

    Attributes
    ----------
    endog : ndarray
        See Parameters.
    exog : ndarray
        See Parameters.
    """
    _formula_max_endog = None

    def __init__(self, endog, exog, missing='none', hasconst=None, **kwargs):
        if len(endog.shape) == 1 or endog.shape[1] == 1:
            raise ValueError('There must be more than one dependent variable to fit multivariate OLS!')
        super().__init__(endog, exog, missing=missing, hasconst=hasconst, **kwargs)

    def fit(self, method='svd'):
        self._fittedmod = _multivariate_ols_fit(self.endog, self.exog, method=method)
        return _MultivariateOLSResults(self)