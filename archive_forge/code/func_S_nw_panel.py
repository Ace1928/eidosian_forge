import numpy as np
from statsmodels.tools.grouputils import combine_indices, group_sums
from statsmodels.stats.moment_helpers import se_cov
def S_nw_panel(xw, weights, groupidx):
    """inner covariance matrix for HAC for panel data

    no denominator nobs used

    no reference for this, just accounting for time indices
    """
    nlags = len(weights) - 1
    S = weights[0] * np.dot(xw.T, xw)
    for lag in range(1, nlags + 1):
        xw0, xwlag = lagged_groups(xw, lag, groupidx)
        s = np.dot(xw0.T, xwlag)
        S += weights[lag] * (s + s.T)
    return S