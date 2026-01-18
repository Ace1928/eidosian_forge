import numpy as np
import scipy.special
import scipy.special._ufuncs as scu
from scipy._lib._finite_differences import _derivative
def _pomeranz_compute_j1j2(i, n, ll, ceilf, roundf):
    """Compute the endpoints of the interval for row i."""
    if i == 0:
        j1, j2 = (-ll - ceilf - 1, ll + ceilf - 1)
    else:
        ip1div2, ip1mod2 = divmod(i + 1, 2)
        if ip1mod2 == 0:
            if ip1div2 == n + 1:
                j1, j2 = (n - ll - ceilf - 1, n + ll + ceilf - 1)
            else:
                j1, j2 = (ip1div2 - 1 - ll - roundf - 1, ip1div2 + ll - 1 + ceilf - 1)
        else:
            j1, j2 = (ip1div2 - 1 - ll - 1, ip1div2 + ll + roundf - 1)
    return (max(j1 + 2, 0), min(j2, n))