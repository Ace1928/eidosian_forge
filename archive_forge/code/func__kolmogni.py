import numpy as np
import scipy.special
import scipy.special._ufuncs as scu
from scipy._lib._finite_differences import _derivative
def _kolmogni(n, p, q):
    """Computes the PPF/ISF of kolmogn.

    n of type integer, n>= 1
    p is the CDF, q the SF, p+q=1
    """
    if np.isnan(n):
        return n
    if int(n) != n or n <= 0:
        return np.nan
    if p <= 0:
        return 1.0 / n
    if q <= 0:
        return 1.0
    delta = np.exp((np.log(p) - scipy.special.loggamma(n + 1)) / n)
    if delta <= 1.0 / n:
        return (delta + 1.0 / n) / 2
    x = -np.expm1(np.log(q / 2.0) / n)
    if x >= 1 - 1.0 / n:
        return x
    x1 = scu._kolmogci(p) / np.sqrt(n)
    x1 = min(x1, 1.0 - 1.0 / n)

    def _f(x):
        return _kolmogn(n, x) - p
    return scipy.optimize.brentq(_f, 1.0 / n, x1, xtol=1e-14)