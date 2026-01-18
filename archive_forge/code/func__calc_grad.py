from statsmodels.base.elastic_net import RegularizedResults
from statsmodels.stats.regularized_covariance import _calc_nodewise_row, \
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.regression.linear_model import OLS
import numpy as np
def _calc_grad(mod, params, alpha, L1_wt, score_kwds):
    """calculates the log-likelihood gradient for the debiasing

    Parameters
    ----------
    mod : statsmodels model class instance
        The model for the current partition.
    params : array_like
        The estimated coefficients for the current partition.
    alpha : scalar or array_like
        The penalty weight.  If a scalar, the same penalty weight
        applies to all variables in the model.  If a vector, it
        must have the same length as `params`, and contains a
        penalty weight for each coefficient.
    L1_wt : scalar
        The fraction of the penalty given to the L1 penalty term.
        Must be between 0 and 1 (inclusive).  If 0, the fit is
        a ridge fit, if 1 it is a lasso fit.
    score_kwds : dict-like or None
        Keyword arguments for the score function.

    Returns
    -------
    An array-like object of the same dimension as params

    Notes
    -----
    In general:

    gradient l_k(params)

    where k corresponds to the index of the partition

    For OLS:

    X^T(y - X^T params)
    """
    grad = -mod.score(np.asarray(params), **score_kwds)
    grad += alpha * (1 - L1_wt)
    return grad