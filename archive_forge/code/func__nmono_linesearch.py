import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import svds
from scipy.optimize import fminbound
import warnings
from statsmodels.tools.tools import Bunch
from statsmodels.tools.sm_exceptions import (
def _nmono_linesearch(obj, grad, x, d, obj_hist, M=10, sig1=0.1, sig2=0.9, gam=0.0001, maxiter=100):
    """
    Implements the non-monotone line search of Grippo et al. (1986),
    as described in Birgin, Martinez and Raydan (2013).

    Parameters
    ----------
    obj : real-valued function
        The objective function, to be minimized
    grad : vector-valued function
        The gradient of the objective function
    x : array_like
        The starting point for the line search
    d : array_like
        The search direction
    obj_hist : array_like
        Objective function history (must contain at least one value)
    M : positive int
        Number of previous function points to consider (see references
        for details).
    sig1 : real
        Tuning parameter, see references for details.
    sig2 : real
        Tuning parameter, see references for details.
    gam : real
        Tuning parameter, see references for details.
    maxiter : int
        The maximum number of iterations; returns Nones if convergence
        does not occur by this point

    Returns
    -------
    alpha : real
        The step value
    x : Array_like
        The function argument at the final step
    obval : Real
        The function value at the final step
    g : Array_like
        The gradient at the final step

    Notes
    -----
    The basic idea is to take a big step in the direction of the
    gradient, even if the function value is not decreased (but there
    is a maximum allowed increase in terms of the recent history of
    the iterates).

    References
    ----------
    Grippo L, Lampariello F, Lucidi S (1986). A Nonmonotone Line
    Search Technique for Newton's Method. SIAM Journal on Numerical
    Analysis, 23, 707-716.

    E. Birgin, J.M. Martinez, and M. Raydan. Spectral projected
    gradient methods: Review and perspectives. Journal of Statistical
    Software (preprint).
    """
    alpha = 1.0
    last_obval = obj(x)
    obj_max = max(obj_hist[-M:])
    for iter in range(maxiter):
        obval = obj(x + alpha * d)
        g = grad(x)
        gtd = (g * d).sum()
        if obval <= obj_max + gam * alpha * gtd:
            return (alpha, x + alpha * d, obval, g)
        a1 = -0.5 * alpha ** 2 * gtd / (obval - last_obval - alpha * gtd)
        if sig1 <= a1 and a1 <= sig2 * alpha:
            alpha = a1
        else:
            alpha /= 2.0
        last_obval = obval
    return (None, None, None, None)