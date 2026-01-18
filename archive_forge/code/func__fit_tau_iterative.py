import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.base import HolderTuple
def _fit_tau_iterative(eff, var_eff, tau2_start=0, atol=1e-05, maxiter=50):
    """Paule-Mandel iterative estimate of between random effect variance

    implementation follows DerSimonian and Kacker 2007 Appendix 8
    see also Kacker 2004

    Parameters
    ----------
    eff : ndarray
        effect sizes
    var_eff : ndarray
        variance of effect sizes
    tau2_start : float
        starting value for iteration
    atol : float, default: 1e-5
        convergence tolerance for absolute value of estimating equation
    maxiter : int
        maximum number of iterations

    Returns
    -------
    tau2 : float
        estimate of random effects variance tau squared
    converged : bool
        True if iteration has converged.

    """
    tau2 = tau2_start
    k = eff.shape[0]
    converged = False
    for i in range(maxiter):
        w = 1 / (var_eff + tau2)
        m = w.dot(eff) / w.sum(0)
        resid_sq = (eff - m) ** 2
        q_w = w.dot(resid_sq)
        ee = q_w - (k - 1)
        if ee < 0:
            tau2 = 0
            converged = 0
            break
        if np.allclose(ee, 0, atol=atol):
            converged = True
            break
        delta = ee / (w ** 2).dot(resid_sq)
        tau2 += delta
    return (tau2, converged)