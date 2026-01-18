from warnings import warn
from scipy.optimize import _minpack2 as minpack2    # noqa: F401
from ._dcsrch import DCSRCH
import numpy as np
def _zoom(a_lo, a_hi, phi_lo, phi_hi, derphi_lo, phi, derphi, phi0, derphi0, c1, c2, extra_condition):
    """Zoom stage of approximate linesearch satisfying strong Wolfe conditions.

    Part of the optimization algorithm in `scalar_search_wolfe2`.

    Notes
    -----
    Implements Algorithm 3.6 (zoom) in Wright and Nocedal,
    'Numerical Optimization', 1999, pp. 61.

    """
    maxiter = 10
    i = 0
    delta1 = 0.2
    delta2 = 0.1
    phi_rec = phi0
    a_rec = 0
    while True:
        dalpha = a_hi - a_lo
        if dalpha < 0:
            a, b = (a_hi, a_lo)
        else:
            a, b = (a_lo, a_hi)
        if i > 0:
            cchk = delta1 * dalpha
            a_j = _cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec)
        if i == 0 or a_j is None or a_j > b - cchk or (a_j < a + cchk):
            qchk = delta2 * dalpha
            a_j = _quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
            if a_j is None or a_j > b - qchk or a_j < a + qchk:
                a_j = a_lo + 0.5 * dalpha
        phi_aj = phi(a_j)
        if phi_aj > phi0 + c1 * a_j * derphi0 or phi_aj >= phi_lo:
            phi_rec = phi_hi
            a_rec = a_hi
            a_hi = a_j
            phi_hi = phi_aj
        else:
            derphi_aj = derphi(a_j)
            if abs(derphi_aj) <= -c2 * derphi0 and extra_condition(a_j, phi_aj):
                a_star = a_j
                val_star = phi_aj
                valprime_star = derphi_aj
                break
            if derphi_aj * (a_hi - a_lo) >= 0:
                phi_rec = phi_hi
                a_rec = a_hi
                a_hi = a_lo
                phi_hi = phi_lo
            else:
                phi_rec = phi_lo
                a_rec = a_lo
            a_lo = a_j
            phi_lo = phi_aj
            derphi_lo = derphi_aj
        i += 1
        if i > maxiter:
            a_star = None
            val_star = None
            valprime_star = None
            break
    return (a_star, val_star, valprime_star)