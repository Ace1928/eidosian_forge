import numpy as np
from statsmodels.tools.grouputils import combine_indices, group_sums
from statsmodels.stats.moment_helpers import se_cov
def S_hac_simple(x, nlags=None, weights_func=weights_bartlett):
    """inner covariance matrix for HAC (Newey, West) sandwich

    assumes we have a single time series with zero axis consecutive, equal
    spaced time periods


    Parameters
    ----------
    x : ndarray (nobs,) or (nobs, k_var)
        data, for HAC this is array of x_i * u_i
    nlags : int or None
        highest lag to include in kernel window. If None, then
        nlags = floor(4(T/100)^(2/9)) is used.
    weights_func : callable
        weights_func is called with nlags as argument to get the kernel
        weights. default are Bartlett weights

    Returns
    -------
    S : ndarray, (k_vars, k_vars)
        inner covariance matrix for sandwich

    Notes
    -----
    used by cov_hac_simple

    options might change when other kernels besides Bartlett are available.

    """
    if x.ndim == 1:
        x = x[:, None]
    n_periods = x.shape[0]
    if nlags is None:
        nlags = int(np.floor(4 * (n_periods / 100.0) ** (2.0 / 9.0)))
    weights = weights_func(nlags)
    S = weights[0] * np.dot(x.T, x)
    for lag in range(1, nlags + 1):
        s = np.dot(x[lag:].T, x[:-lag])
        S += weights[lag] * (s + s.T)
    return S