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
@deprecate_kwarg('maxlag', 'nlags')
def acorr_lm(resid, nlags=None, store=False, *, period=None, ddof=0, cov_type='nonrobust', cov_kwargs=None):
    """
    Lagrange Multiplier tests for autocorrelation.

    This is a generic Lagrange Multiplier test for autocorrelation. Returns
    Engle's ARCH test if resid is the squared residual array. Breusch-Godfrey
    is a variation on this test with additional exogenous variables.

    Parameters
    ----------
    resid : array_like
        Time series to test.
    nlags : int, default None
        Highest lag to use.
    store : bool, default False
        If true then the intermediate results are also returned.
    period : int, default none
        The period of a Seasonal time series.  Used to compute the max lag
        for seasonal data which uses min(2*period, nobs // 5) if set. If None,
        then the default rule is used to set the number of lags. When set, must
        be >= 2.
    ddof : int, default 0
        The number of degrees of freedom consumed by the model used to
        produce resid. The default value is 0.
    cov_type : str, default "nonrobust"
        Covariance type. The default is "nonrobust` which uses the classic
        OLS covariance estimator. Specify one of "HC0", "HC1", "HC2", "HC3"
        to use White's covariance estimator. All covariance types supported
        by ``OLS.fit`` are accepted.
    cov_kwargs : dict, default None
        Dictionary of covariance options passed to ``OLS.fit``. See OLS.fit for
        more details.

    Returns
    -------
    lm : float
        Lagrange multiplier test statistic.
    lmpval : float
        The p-value for Lagrange multiplier test.
    fval : float
        The f statistic of the F test, alternative version of the same
        test based on F test for the parameter restriction.
    fpval : float
        The pvalue of the F test.
    res_store : ResultsStore, optional
        Intermediate results. Only returned if store=True.

    See Also
    --------
    het_arch
        Conditional heteroskedasticity testing.
    acorr_breusch_godfrey
        Breusch-Godfrey test for serial correlation.
    acorr_ljung_box
        Ljung-Box test for serial correlation.

    Notes
    -----
    The test statistic is computed as (nobs - ddof) * r2 where r2 is the
    R-squared from a regression on the residual on nlags lags of the
    residual.
    """
    resid = array_like(resid, 'resid', ndim=1)
    cov_type = string_like(cov_type, 'cov_type')
    cov_kwargs = {} if cov_kwargs is None else cov_kwargs
    cov_kwargs = dict_like(cov_kwargs, 'cov_kwargs')
    nobs = resid.shape[0]
    if period is not None and nlags is None:
        maxlag = min(nobs // 5, 2 * period)
    elif nlags is None:
        maxlag = min(10, nobs // 5)
    else:
        maxlag = nlags
    xdall = lagmat(resid[:, None], maxlag, trim='both')
    nobs = xdall.shape[0]
    xdall = np.c_[np.ones((nobs, 1)), xdall]
    xshort = resid[-nobs:]
    res_store = ResultsStore()
    usedlag = maxlag
    resols = OLS(xshort, xdall[:, :usedlag + 1]).fit(cov_type=cov_type, cov_kwargs=cov_kwargs)
    fval = float(resols.fvalue)
    fpval = float(resols.f_pvalue)
    if cov_type == 'nonrobust':
        lm = (nobs - ddof) * resols.rsquared
        lmpval = stats.chi2.sf(lm, usedlag)
    else:
        r_matrix = np.hstack((np.zeros((usedlag, 1)), np.eye(usedlag)))
        test_stat = resols.wald_test(r_matrix, use_f=False, scalar=True)
        lm = float(test_stat.statistic)
        lmpval = float(test_stat.pvalue)
    if store:
        res_store.resols = resols
        res_store.usedlag = usedlag
        return (lm, lmpval, fval, fpval, res_store)
    else:
        return (lm, lmpval, fval, fpval)