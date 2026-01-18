import warnings
import math
from math import pi, prod
import cupy
from cupyx.scipy.special import binom as comb
import cupyx.scipy.special as special
from cupyx.scipy.signal import _optimize
from cupyx.scipy.signal._polyutils import roots, poly
from cupyx.scipy.signal._lti_conversion import abcd_normalize
def _ellipdeg(n, m1):
    """Solve degree equation using nomes

    Given n, m1, solve
       n * K(m) / K'(m) = K1(m1) / K1'(m1)
    for m

    See [1], Eq. (49)

    References
    ----------
    .. [1] Orfanidis, "Lecture Notes on Elliptic Filter Design",
           https://www.ece.rutgers.edu/~orfanidi/ece521/notes.pdf
    """
    _ELLIPDEG_MMAX = 7
    K1 = special.ellipk(m1)
    K1p = special.ellipkm1(m1)
    q1 = cupy.exp(-pi * K1p / K1)
    q = q1 ** (1 / n)
    mnum = cupy.arange(_ELLIPDEG_MMAX + 1)
    mden = cupy.arange(1, _ELLIPDEG_MMAX + 2)
    num = (q ** (mnum * (mnum + 1))).sum()
    den = 1 + 2 * (q ** mden ** 2).sum()
    return 16 * q * (num / den) ** 4