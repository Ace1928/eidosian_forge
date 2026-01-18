import warnings
import numpy as np
from scipy import stats
from statsmodels.tools.validation import array_like, bool_like, int_like

    Anderson-Darling test for normal distribution unknown mean and variance.

    Parameters
    ----------
    x : array_like
        The data array.
    axis : int
        The axis to perform the test along.

    Returns
    -------
    ad2 : float
        Anderson Darling test statistic.
    pval : float
        The pvalue for hypothesis that the data comes from a normal
        distribution with unknown mean and variance.

    See Also
    --------
    statsmodels.stats.diagnostic.anderson_statistic
        The Anderson-Darling a2 statistic.
    statsmodels.stats.diagnostic.kstest_fit
        Kolmogorov-Smirnov test with estimated parameters for Normal or
        Exponential distributions.
    