import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.base import HolderTuple
def _fit_tau_mm(eff, var_eff, weights):
    """one-step method of moment estimate of between random effect variance

    implementation follows Kacker 2004 and DerSimonian and Kacker 2007 eq. 6

    Parameters
    ----------
    eff : ndarray
        effect sizes
    var_eff : ndarray
        variance of effect sizes
    weights : ndarray
        weights for estimating overall weighted mean

    Returns
    -------
    tau2 : float
        estimate of random effects variance tau squared

    """
    w = weights
    m = w.dot(eff) / w.sum(0)
    resid_sq = (eff - m) ** 2
    q_w = w.dot(resid_sq)
    w_t = w.sum()
    expect = w.dot(var_eff) - (w ** 2).dot(var_eff) / w_t
    denom = w_t - (w ** 2).sum() / w_t
    tau2 = (q_w - expect) / denom
    return tau2