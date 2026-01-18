import numpy as np
from numpy import (exp, inf, pi, sqrt, floor, sin, cos, around,
from scipy import linalg
from scipy.special import airy
from . import _specfun  # type: ignore
from . import _ufuncs
def _newton(n, x_initial, maxit=5):
    """Newton iteration for polishing the asymptotic approximation
    to the zeros of the Hermite polynomials.

    Parameters
    ----------
    n : int
        Quadrature order
    x_initial : ndarray
        Initial guesses for the roots
    maxit : int
        Maximal number of Newton iterations.
        The default 5 is sufficient, usually
        only one or two steps are needed.

    Returns
    -------
    nodes : ndarray
        Quadrature nodes
    weights : ndarray
        Quadrature weights

    See Also
    --------
    roots_hermite_asy
    """
    mu = sqrt(2.0 * n + 1.0)
    t = x_initial / mu
    theta = arccos(t)
    for i in range(maxit):
        u, ud = _pbcf(n, theta)
        dtheta = u / (sqrt(2.0) * mu * sin(theta) * ud)
        theta = theta + dtheta
        if max(abs(dtheta)) < 1e-14:
            break
    x = mu * cos(theta)
    if n % 2 == 1:
        x[0] = 0.0
    w = exp(-x ** 2) / (2.0 * ud ** 2)
    return (x, w)