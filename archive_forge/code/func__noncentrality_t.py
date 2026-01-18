import numpy as np
from scipy import special
from statsmodels.stats.base import Holder
def _noncentrality_t(t_stat, df, alpha=0.05):
    """noncentrality parameter for t statistic

    Parameters
    ----------
    fstat : float
        f-statistic, for example from a hypothesis test
        df : int or float
        Degrees of freedom
    alpha : float in (0, 1)
        Significance level for the confidence interval, covarage is 1 - alpha.

    Returns
    -------
    HolderTuple
        The main attributes are

        - ``nc`` : estimate of noncentrality parameter
        - ``confint`` : lower and upper bound of confidence interval for `nc``

        Other attributes are estimates for nc by different methods.

    References
    ----------
    .. [1] Hedges, Larry V. 2016. “Distribution Theory for Glass’s Estimator of
       Effect Size and Related Estimators:”
       Journal of Educational Statistics, November.
       https://doi.org/10.3102/10769986006002107.

    """
    alpha_half = alpha / 2
    gfac = np.exp(special.gammaln(df / 2.0 - 0.5) - special.gammaln(df / 2.0))
    c11 = np.sqrt(df / 2.0) * gfac
    nc = t_stat / c11
    nc_median = special.nctdtrinc(df, 0.5, t_stat)
    ci = special.nctdtrinc(df, [1 - alpha_half, alpha_half], t_stat)
    res = Holder(nc=nc, confint=ci, nc_median=nc_median, name='Noncentrality for t-distributed random variable')
    return res