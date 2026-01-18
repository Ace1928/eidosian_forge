from statsmodels.regression.linear_model import OLS
import numpy as np
def _calc_nodewise_weight(exog, nodewise_row, idx, alpha):
    """calculates the nodewise_weightvalue for the idxth variable, used to
    estimate approx_inv_cov.

    Parameters
    ----------
    exog : array_like
        The weighted design matrix for the current partition.
    nodewise_row : array_like
        The nodewise_row values for the current variable.
    idx : scalar
        Index of the current variable
    alpha : scalar or array_like
        The penalty weight.  If a scalar, the same penalty weight
        applies to all variables in the model.  If a vector, it
        must have the same length as `params`, and contains a
        penalty weight for each coefficient.

    Returns
    -------
    A scalar

    Notes
    -----

    nodewise_weight_i = sqrt(1/n ||exog,i - exog_-i nodewise_row||_2^2
                             + alpha ||nodewise_row||_1)
    """
    n, p = exog.shape
    ind = list(range(p))
    ind.pop(idx)
    if not np.isscalar(alpha):
        alpha = alpha[ind]
    d = np.linalg.norm(exog[:, idx] - exog[:, ind].dot(nodewise_row)) ** 2
    d = np.sqrt(d / n + alpha * np.linalg.norm(nodewise_row, 1))
    return d