from statsmodels.compat.pandas import deprecate_kwarg
from collections.abc import Iterable
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS, RegressionResultsWrapper
from statsmodels.stats._adnorm import anderson_statistic, normal_ad
from statsmodels.stats._lilliefors import (
from statsmodels.tools.validation import (
from statsmodels.tsa.tsatools import lagmat
def acorr_ljungbox(x, lags=None, boxpierce=False, model_df=0, period=None, return_df=True, auto_lag=False):
    """
    Ljung-Box test of autocorrelation in residuals.

    Parameters
    ----------
    x : array_like
        The data series. The data is demeaned before the test statistic is
        computed.
    lags : {int, array_like}, default None
        If lags is an integer then this is taken to be the largest lag
        that is included, the test result is reported for all smaller lag
        length. If lags is a list or array, then all lags are included up to
        the largest lag in the list, however only the tests for the lags in
        the list are reported. If lags is None, then the default maxlag is
        min(10, nobs // 5). The default number of lags changes if period
        is set.
    boxpierce : bool, default False
        If true, then additional to the results of the Ljung-Box test also the
        Box-Pierce test results are returned.
    model_df : int, default 0
        Number of degrees of freedom consumed by the model. In an ARMA model,
        this value is usually p+q where p is the AR order and q is the MA
        order. This value is subtracted from the degrees-of-freedom used in
        the test so that the adjusted dof for the statistics are
        lags - model_df. If lags - model_df <= 0, then NaN is returned.
    period : int, default None
        The period of a Seasonal time series.  Used to compute the max lag
        for seasonal data which uses min(2*period, nobs // 5) if set. If None,
        then the default rule is used to set the number of lags. When set, must
        be >= 2.
    auto_lag : bool, default False
        Flag indicating whether to automatically determine the optimal lag
        length based on threshold of maximum correlation value.

    Returns
    -------
    DataFrame
        Frame with columns:

        * lb_stat - The Ljung-Box test statistic.
        * lb_pvalue - The p-value based on chi-square distribution. The
          p-value is computed as 1 - chi2.cdf(lb_stat, dof) where dof is
          lag - model_df. If lag - model_df <= 0, then NaN is returned for
          the pvalue.
        * bp_stat - The Box-Pierce test statistic.
        * bp_pvalue - The p-value based for Box-Pierce test on chi-square
          distribution. The p-value is computed as 1 - chi2.cdf(bp_stat, dof)
          where dof is lag - model_df. If lag - model_df <= 0, then NaN is
          returned for the pvalue.

    See Also
    --------
    statsmodels.regression.linear_model.OLS.fit
        Regression model fitting.
    statsmodels.regression.linear_model.RegressionResults
        Results from linear regression models.
    statsmodels.stats.stattools.q_stat
        Ljung-Box test statistic computed from estimated
        autocorrelations.

    Notes
    -----
    Ljung-Box and Box-Pierce statistic differ in their scaling of the
    autocorrelation function. Ljung-Box test is has better finite-sample
    properties.

    References
    ----------
    .. [*] Green, W. "Econometric Analysis," 5th ed., Pearson, 2003.
    .. [*] J. Carlos Escanciano, Ignacio N. Lobato
          "An automatic Portmanteau test for serial correlation".,
          Volume 151, 2009.

    Examples
    --------
    >>> import statsmodels.api as sm
    >>> data = sm.datasets.sunspots.load_pandas().data
    >>> res = sm.tsa.ARMA(data["SUNACTIVITY"], (1,1)).fit(disp=-1)
    >>> sm.stats.acorr_ljungbox(res.resid, lags=[10], return_df=True)
           lb_stat     lb_pvalue
    10  214.106992  1.827374e-40
    """
    from statsmodels.tsa.stattools import acf
    x = array_like(x, 'x')
    period = int_like(period, 'period', optional=True)
    model_df = int_like(model_df, 'model_df', optional=False)
    if period is not None and period <= 1:
        raise ValueError('period must be >= 2')
    if model_df < 0:
        raise ValueError('model_df must be >= 0')
    nobs = x.shape[0]
    if auto_lag:
        maxlag = nobs - 1
        sacf = acf(x, nlags=maxlag, fft=False)
        if not boxpierce:
            q_sacf = nobs * (nobs + 2) * np.cumsum(sacf[1:maxlag + 1] ** 2 / (nobs - np.arange(1, maxlag + 1)))
        else:
            q_sacf = nobs * np.cumsum(sacf[1:maxlag + 1] ** 2)
        q = 2.4
        threshold = np.sqrt(q * np.log(nobs))
        threshold_metric = np.abs(sacf).max() * np.sqrt(nobs)
        if threshold_metric <= threshold:
            q_sacf = q_sacf - np.arange(1, nobs) * np.log(nobs)
        else:
            q_sacf = q_sacf - 2 * np.arange(1, nobs)
        lags = np.argmax(q_sacf)
        lags = max(1, lags)
        lags = int_like(lags, 'lags')
        lags = np.arange(1, lags + 1)
    elif period is not None:
        lags = np.arange(1, min(nobs // 5, 2 * period) + 1, dtype=int)
    elif lags is None:
        lags = np.arange(1, min(nobs // 5, 10) + 1, dtype=int)
    elif not isinstance(lags, Iterable):
        lags = int_like(lags, 'lags')
        lags = np.arange(1, lags + 1)
    lags = array_like(lags, 'lags', dtype='int')
    maxlag = lags.max()
    sacf = acf(x, nlags=maxlag, fft=False)
    sacf2 = sacf[1:maxlag + 1] ** 2 / (nobs - np.arange(1, maxlag + 1))
    qljungbox = nobs * (nobs + 2) * np.cumsum(sacf2)[lags - 1]
    adj_lags = lags - model_df
    pval = np.full_like(qljungbox, np.nan)
    loc = adj_lags > 0
    pval[loc] = stats.chi2.sf(qljungbox[loc], adj_lags[loc])
    if not boxpierce:
        return pd.DataFrame({'lb_stat': qljungbox, 'lb_pvalue': pval}, index=lags)
    qboxpierce = nobs * np.cumsum(sacf[1:maxlag + 1] ** 2)[lags - 1]
    pvalbp = np.full_like(qljungbox, np.nan)
    pvalbp[loc] = stats.chi2.sf(qboxpierce[loc], adj_lags[loc])
    return pd.DataFrame({'lb_stat': qljungbox, 'lb_pvalue': pval, 'bp_stat': qboxpierce, 'bp_pvalue': pvalbp}, index=lags)