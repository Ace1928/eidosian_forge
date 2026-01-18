import warnings
import numpy as np
from scipy import interpolate, stats
def _ecdf_mv(data, method='seq', use_ranks=True):
    """
    Multivariate empiricial distribution function, empirical copula


    Notes
    -----
    Method "seq" is faster than method "brute", but supports mainly bivariate
    case. Speed advantage of "seq" is increasing in number of observations
    and decreasing in number of variables.
    (see Segers ...)

    Warning: This does not handle ties. The ecdf is based on univariate ranks
    without ties. The assignment of ranks to ties depends on the sorting
    algorithm and the initial ordering of the data.

    When the original data is used instead of ranks, then method "brute"
    computes the correct ecdf counts even in the case of ties.

    """
    x = np.asarray(data)
    n = x.shape[0]
    if use_ranks:
        x = _rankdata_no_ties(x) / n
    if method == 'brute':
        count = [(x <= x[i]).all(1).sum() for i in range(n)]
        count = np.asarray(count)
    elif method.startswith('seq'):
        sort_idx0 = np.argsort(x[:, 0])
        x_s0 = x[sort_idx0]
        x1 = x_s0[:, 1:]
        count_smaller = [(x1[:i] <= x1[i]).all(1).sum() + 1 for i in range(n)]
        count = np.empty(x.shape[0])
        count[sort_idx0] = count_smaller
    else:
        raise ValueError('method not available')
    return (count, x)