from scipy import stats
import numpy as np
from statsmodels.tools.sm_exceptions import ValueWarning
def _medcouple_1d(y):
    """
    Calculates the medcouple robust measure of skew.

    Parameters
    ----------
    y : array_like, 1-d
        Data to compute use in the estimator.

    Returns
    -------
    mc : float
        The medcouple statistic

    Notes
    -----
    The current algorithm requires a O(N**2) memory allocations, and so may
    not work for very large arrays (N>10000).

    .. [*] M. Hubert and E. Vandervieren, "An adjusted boxplot for skewed
       distributions" Computational Statistics & Data Analysis, vol. 52, pp.
       5186-5201, August 2008.
    """
    y = np.squeeze(np.asarray(y))
    if y.ndim != 1:
        raise ValueError('y must be squeezable to a 1-d array')
    y = np.sort(y)
    n = y.shape[0]
    if n % 2 == 0:
        mf = (y[n // 2 - 1] + y[n // 2]) / 2
    else:
        mf = y[(n - 1) // 2]
    z = y - mf
    lower = z[z <= 0.0]
    upper = z[z >= 0.0]
    upper = upper[:, None]
    standardization = upper - lower
    is_zero = np.logical_and(lower == 0.0, upper == 0.0)
    standardization[is_zero] = np.inf
    spread = upper + lower
    h = spread / standardization
    num_ties = np.sum(lower == 0.0)
    if num_ties:
        replacements = np.ones((num_ties, num_ties)) - np.eye(num_ties)
        replacements -= 2 * np.triu(replacements)
        replacements = np.fliplr(replacements)
        h[:num_ties, -num_ties:] = replacements
    return np.median(h)