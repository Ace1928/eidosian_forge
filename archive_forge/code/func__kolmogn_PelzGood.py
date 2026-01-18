import numpy as np
import scipy.special
import scipy.special._ufuncs as scu
from scipy._lib._finite_differences import _derivative
def _kolmogn_PelzGood(n, x, cdf=True):
    """Computes the Pelz-Good approximation to Prob(Dn <= x) with 0<=x<=1.

    Start with Li-Chien, Korolyuk approximation:
        Prob(Dn <= x) ~ K0(z) + K1(z)/sqrt(n) + K2(z)/n + K3(z)/n**1.5
    where z = x*sqrt(n).
    Transform each K_(z) using Jacobi theta functions into a form suitable
    for small z.
    Pelz-Good (1976). [6]
    """
    if x <= 0.0:
        return _select_and_clip_prob(0.0, 1.0, cdf=cdf)
    if x >= 1.0:
        return _select_and_clip_prob(1.0, 0.0, cdf=cdf)
    z = np.sqrt(n) * x
    zsquared, zthree, zfour, zsix = (z ** 2, z ** 3, z ** 4, z ** 6)
    qlog = -_PI_SQUARED / 8 / zsquared
    if qlog < _MIN_LOG:
        return _select_and_clip_prob(0.0, 1.0, cdf=cdf)
    q = np.exp(qlog)
    k1a = -zsquared
    k1b = _PI_SQUARED / 4
    k2a = 6 * zsix + 2 * zfour
    k2b = (2 * zfour - 5 * zsquared) * _PI_SQUARED / 4
    k2c = _PI_FOUR * (1 - 2 * zsquared) / 16
    k3d = _PI_SIX * (5 - 30 * zsquared) / 64
    k3c = _PI_FOUR * (-60 * zsquared + 212 * zfour) / 16
    k3b = _PI_SQUARED * (135 * zfour - 96 * zsix) / 4
    k3a = -30 * zsix - 90 * z ** 8
    K0to3 = np.zeros(4)
    maxk = int(np.ceil(16 * z / np.pi))
    for k in range(maxk, 0, -1):
        m = 2 * k - 1
        msquared, mfour, msix = (m ** 2, m ** 4, m ** 6)
        qpower = np.power(q, 8 * k)
        coeffs = np.array([1.0, k1a + k1b * msquared, k2a + k2b * msquared + k2c * mfour, k3a + k3b * msquared + k3c * mfour + k3d * msix])
        K0to3 *= qpower
        K0to3 += coeffs
    K0to3 *= q
    K0to3 *= _SQRT2PI
    K0to3 /= np.array([z, 6 * zfour, 72 * z ** 7, 6480 * z ** 10])
    q = np.exp(-_PI_SQUARED / 2 / zsquared)
    ks = np.arange(maxk, 0, -1)
    ksquared = ks ** 2
    sqrt3z = _SQRT3 * z
    kspi = np.pi * ks
    qpwers = q ** ksquared
    k2extra = np.sum(ksquared * qpwers)
    k2extra *= _PI_SQUARED * _SQRT2PI / (-36 * zthree)
    K0to3[2] += k2extra
    k3extra = np.sum((sqrt3z + kspi) * (sqrt3z - kspi) * ksquared * qpwers)
    k3extra *= _PI_SQUARED * _SQRT2PI / (216 * zsix)
    K0to3[3] += k3extra
    powers_of_n = np.power(n * 1.0, np.arange(len(K0to3)) / 2.0)
    K0to3 /= powers_of_n
    if not cdf:
        K0to3 *= -1
        K0to3[0] += 1
    Ksum = sum(K0to3)
    return Ksum