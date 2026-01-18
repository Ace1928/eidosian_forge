from statsmodels.base.elastic_net import RegularizedResults
from statsmodels.stats.regularized_covariance import _calc_nodewise_row, \
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.regression.linear_model import OLS
import numpy as np
def _calc_wdesign_mat(mod, params, hess_kwds):
    """calculates the weighted design matrix necessary to generate
    the approximate inverse covariance matrix

    Parameters
    ----------
    mod : statsmodels model class instance
        The model for the current partition.
    params : array_like
        The estimated coefficients for the current partition.
    hess_kwds : dict-like or None
        Keyword arguments for the hessian function.

    Returns
    -------
    An array-like object, updated design matrix, same dimension
    as mod.exog
    """
    rhess = np.sqrt(mod.hessian_factor(np.asarray(params), **hess_kwds))
    return rhess[:, None] * mod.exog