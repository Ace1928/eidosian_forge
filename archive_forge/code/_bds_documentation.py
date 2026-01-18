import numpy as np
from scipy import stats
from statsmodels.tools.validation import array_like

    BDS Test Statistic for Independence of a Time Series

    Parameters
    ----------
    x : ndarray
        Observations of time series for which bds statistics is calculated.
    max_dim : int
        The maximum embedding dimension.
    epsilon : {float, None}, optional
        The threshold distance to use in calculating the correlation sum.
    distance : float, optional
        Specifies the distance multiplier to use when computing the test
        statistic if epsilon is omitted.

    Returns
    -------
    bds_stat : float
        The BDS statistic.
    pvalue : float
        The p-values associated with the BDS statistic.

    Notes
    -----
    The null hypothesis of the test statistic is for an independent and
    identically distributed (i.i.d.) time series, and an unspecified
    alternative hypothesis.

    This test is often used as a residual diagnostic.

    The calculation involves matrices of size (nobs, nobs), so this test
    will not work with very long datasets.

    Implementation conditions on the first m-1 initial values, which are
    required to calculate the m-histories:
    x_t^m = (x_t, x_{t-1}, ... x_{t-(m-1)})
    