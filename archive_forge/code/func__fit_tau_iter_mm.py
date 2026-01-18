import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.base import HolderTuple
def _fit_tau_iter_mm(eff, var_eff, tau2_start=0, atol=1e-05, maxiter=50):
    """iterated method of moment estimate of between random effect variance

    This repeatedly estimates tau, updating weights in each iteration
    see two-step estimators in DerSimonian and Kacker 2007

    Parameters
    ----------
    eff : ndarray
        effect sizes
    var_eff : ndarray
        variance of effect sizes
    tau2_start : float
        starting value for iteration
    atol : float, default: 1e-5
        convergence tolerance for change in tau2 estimate between iterations
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
    converged = False
    for _ in range(maxiter):
        w = 1 / (var_eff + tau2)
        tau2_new = _fit_tau_mm(eff, var_eff, w)
        tau2_new = max(0, tau2_new)
        delta = tau2_new - tau2
        if np.allclose(delta, 0, atol=atol):
            converged = True
            break
        tau2 = tau2_new
    return (tau2, converged)