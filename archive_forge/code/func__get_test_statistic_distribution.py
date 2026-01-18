from collections import namedtuple
import warnings
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
def _get_test_statistic_distribution(x, y, B):
    """
    Parameters
    ----------
    x : array_like, 1-D or 2-D
        If `x` is 1-D than it is assumed to be a vector of observations of a
        single random variable. If `x` is 2-D than the rows should be
        observations and the columns are treated as the components of a
        random vector, i.e., each column represents a different component of
        the random vector `x`.
    y : array_like, 1-D or 2-D
        Same as `x`, but only the number of observation has to match that of
        `x`. If `y` is 2-D note that the number of columns of `y` (i.e., the
        number of components in the random vector) does not need to match
        the number of columns in `x`.
    B : int
        The number of iterations to perform when evaluating the null
        distribution.

    Returns
    -------
    emp_dist : array_like
        The empirical distribution of the test statistic.

    """
    y = y.copy()
    emp_dist = np.zeros(B)
    x_dist = squareform(pdist(x, 'euclidean'))
    for i in range(B):
        np.random.shuffle(y)
        emp_dist[i] = distance_statistics(x, y, x_dist=x_dist).test_statistic
    return emp_dist