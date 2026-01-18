from collections import namedtuple
import warnings
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
def _asymptotic_pvalue(stats):
    """Calculate the p-value based on an approximation of the distribution of
    the test statistic under the null.

    Parameters
    ----------
    stats: namedtuple
        The result obtained from calling ``distance_statistics(x, y)``.

    Returns
    -------
    test_statistic : float
        The test statistic.
    pval : float
        The asymptotic p-value.

    """
    test_statistic = np.sqrt(stats.test_statistic / stats.S)
    pval = (1 - norm.cdf(test_statistic)) * 2
    return (test_statistic, pval)