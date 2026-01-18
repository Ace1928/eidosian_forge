import numpy as np
from statsmodels.tools.grouputils import combine_indices, group_sums
from statsmodels.stats.moment_helpers import se_cov
def S_hac_groupsum(x, time, nlags=None, weights_func=weights_bartlett):
    """inner covariance matrix for HAC over group sums sandwich

    This assumes we have complete equal spaced time periods.
    The number of time periods per group need not be the same, but we need
    at least one observation for each time period

    For a single categorical group only, or a everything else but time
    dimension. This first aggregates x over groups for each time period, then
    applies HAC on the sum per period.

    Parameters
    ----------
    x : ndarray (nobs,) or (nobs, k_var)
        data, for HAC this is array of x_i * u_i
    time : ndarray, (nobs,)
        timeindes, assumed to be integers range(n_periods)
    nlags : int or None
        highest lag to include in kernel window. If None, then
        nlags = floor[4(T/100)^(2/9)] is used.
    weights_func : callable
        weights_func is called with nlags as argument to get the kernel
        weights. default are Bartlett weights

    Returns
    -------
    S : ndarray, (k_vars, k_vars)
        inner covariance matrix for sandwich

    References
    ----------
    Daniel Hoechle, xtscc paper
    Driscoll and Kraay

    """
    x_group_sums = group_sums(x, time).T
    return S_hac_simple(x_group_sums, nlags=nlags, weights_func=weights_func)