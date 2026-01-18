import numpy as np
from statsmodels.duration.hazard_regression import PHReg
def _kernel_survfunc(time, status, exog, kfunc, freq_weights):
    """
    Estimate the marginal survival function under dependent censoring.

    Parameters
    ----------
    time : array_like
        The observed times for each subject
    status : array_like
        The status for each subject (1 indicates event, 0 indicates
        censoring)
    exog : array_like
        Covariates such that censoring is independent conditional on
        exog
    kfunc : function
        Kernel function
    freq_weights : array_like
        Optional frequency weights

    Returns
    -------
    probs : array_like
        The estimated survival probabilities
    times : array_like
        The times at which the survival probabilities are estimated

    References
    ----------
    Zeng, Donglin 2004. Estimating Marginal Survival Function by
    Adjusting for Dependent Censoring Using Many Covariates. The
    Annals of Statistics 32 (4): 1533 55.
    doi:10.1214/009053604000000508.
    https://arxiv.org/pdf/math/0409180.pdf
    """
    sfe = PHReg(time, exog, status).fit()
    fitval_e = sfe.predict().predicted_values
    sfc = PHReg(time, exog, 1 - status).fit()
    fitval_c = sfc.predict().predicted_values
    exog2d = np.hstack((fitval_e[:, None], fitval_c[:, None]))
    n = len(time)
    ixd = np.flatnonzero(status == 1)
    utime = np.unique(time[ixd])
    ii = np.argsort(time)
    time = time[ii]
    status = status[ii]
    exog2d = exog2d[ii, :]
    ie = np.searchsorted(time, utime, side='right') - 1
    if freq_weights is not None:
        freq_weights = freq_weights / freq_weights.sum()
    sprob = 0.0
    for i in range(n):
        kd = exog2d - exog2d[i, :]
        kd = kfunc(kd)
        denom = np.cumsum(kd[::-1])[::-1]
        num = kd * status
        rat = num / denom
        tr = 1e-15
        ii = np.flatnonzero((denom < tr) & (num < tr))
        rat[ii] = 0
        ratc = 1 - rat
        ratc = np.clip(ratc, 1e-12, np.inf)
        lrat = np.log(ratc)
        prat = np.cumsum(lrat)[ie]
        prat = np.exp(prat)
        if freq_weights is None:
            sprob += prat
        else:
            sprob += prat * freq_weights[i]
    if freq_weights is None:
        sprob /= n
    return (sprob, utime)