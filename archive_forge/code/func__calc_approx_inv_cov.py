from statsmodels.regression.linear_model import OLS
import numpy as np
def _calc_approx_inv_cov(nodewise_row_l, nodewise_weight_l):
    """calculates the approximate inverse covariance matrix

    Parameters
    ----------
    nodewise_row_l : list
        A list of array-like object where each object corresponds to
        the nodewise_row values for the corresponding variable, should
        be length p.
    nodewise_weight_l : list
        A list of scalars where each scalar corresponds to the nodewise_weight
        value for the corresponding variable, should be length p.

    Returns
    ------
    An array-like object, p x p matrix

    Notes
    -----

    nwr = nodewise_row
    nww = nodewise_weight

    approx_inv_cov_j = - 1 / nww_j [nwr_j,1,...,1,...nwr_j,p]
    """
    p = len(nodewise_weight_l)
    approx_inv_cov = -np.eye(p)
    for idx in range(p):
        ind = list(range(p))
        ind.pop(idx)
        approx_inv_cov[idx, ind] = nodewise_row_l[idx]
    approx_inv_cov *= -1 / nodewise_weight_l[:, None] ** 2
    return approx_inv_cov