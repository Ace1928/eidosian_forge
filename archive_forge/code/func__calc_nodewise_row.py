from statsmodels.regression.linear_model import OLS
import numpy as np
def _calc_nodewise_row(exog, idx, alpha):
    """calculates the nodewise_row values for the idxth variable, used to
    estimate approx_inv_cov.

    Parameters
    ----------
    exog : array_like
        The weighted design matrix for the current partition.
    idx : scalar
        Index of the current variable.
    alpha : scalar or array_like
        The penalty weight.  If a scalar, the same penalty weight
        applies to all variables in the model.  If a vector, it
        must have the same length as `params`, and contains a
        penalty weight for each coefficient.

    Returns
    -------
    An array-like object of length p-1

    Notes
    -----

    nodewise_row_i = arg min 1/(2n) ||exog_i - exog_-i gamma||_2^2
                             + alpha ||gamma||_1
    """
    p = exog.shape[1]
    ind = list(range(p))
    ind.pop(idx)
    if not np.isscalar(alpha):
        alpha = alpha[ind]
    tmod = OLS(exog[:, idx], exog[:, ind])
    nodewise_row = tmod.fit_regularized(alpha=alpha).params
    return nodewise_row