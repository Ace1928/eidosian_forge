from warnings import warn
from scipy.optimize import _minpack2 as minpack2    # noqa: F401
from ._dcsrch import DCSRCH
import numpy as np
def _nonmonotone_line_search_cheng(f, x_k, d, f_k, C, Q, eta, gamma=0.0001, tau_min=0.1, tau_max=0.5, nu=0.85):
    """
    Nonmonotone line search from [1]

    Parameters
    ----------
    f : callable
        Function returning a tuple ``(f, F)`` where ``f`` is the value
        of a merit function and ``F`` the residual.
    x_k : ndarray
        Initial position.
    d : ndarray
        Search direction.
    f_k : float
        Initial merit function value.
    C, Q : float
        Control parameters. On the first iteration, give values
        Q=1.0, C=f_k
    eta : float
        Allowed merit function increase, see [1]_
    nu, gamma, tau_min, tau_max : float, optional
        Search parameters, see [1]_

    Returns
    -------
    alpha : float
        Step length
    xp : ndarray
        Next position
    fp : float
        Merit function value at next position
    Fp : ndarray
        Residual at next position
    C : float
        New value for the control parameter C
    Q : float
        New value for the control parameter Q

    References
    ----------
    .. [1] W. Cheng & D.-H. Li, ''A derivative-free nonmonotone line
           search and its application to the spectral residual
           method'', IMA J. Numer. Anal. 29, 814 (2009).

    """
    alpha_p = 1
    alpha_m = 1
    alpha = 1
    while True:
        xp = x_k + alpha_p * d
        fp, Fp = f(xp)
        if fp <= C + eta - gamma * alpha_p ** 2 * f_k:
            alpha = alpha_p
            break
        alpha_tp = alpha_p ** 2 * f_k / (fp + (2 * alpha_p - 1) * f_k)
        xp = x_k - alpha_m * d
        fp, Fp = f(xp)
        if fp <= C + eta - gamma * alpha_m ** 2 * f_k:
            alpha = -alpha_m
            break
        alpha_tm = alpha_m ** 2 * f_k / (fp + (2 * alpha_m - 1) * f_k)
        alpha_p = np.clip(alpha_tp, tau_min * alpha_p, tau_max * alpha_p)
        alpha_m = np.clip(alpha_tm, tau_min * alpha_m, tau_max * alpha_m)
    Q_next = nu * Q + 1
    C = (nu * Q * (C + eta) + fp) / Q_next
    Q = Q_next
    return (alpha, xp, fp, Fp, C, Q)