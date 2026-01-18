import math
import numpy as np
from scipy import special
from scipy.stats._qmc import primes_from_2_to
def _qsimvtv(m, nu, sigma, a, b, rng):
    """Estimates the multivariate t CDF using randomized QMC

    Parameters
    ----------
    m : int
        The number of points
    nu : float
        Degrees of freedom
    sigma : ndarray
        A 2D positive semidefinite covariance matrix
    a : ndarray
        Lower integration limits
    b : ndarray
        Upper integration limits.
    rng : Generator
        Pseudorandom number generator

    Returns
    -------
    p : float
        The estimated CDF.
    e : float
        An absolute error estimate.

    """
    sn = max(1, math.sqrt(nu))
    ch, az, bz = _chlrps(sigma, a / sn, b / sn)
    n = len(sigma)
    N = 10
    P = math.ceil(m / N)
    on = np.ones(P)
    p = 0
    e = 0
    ps = np.sqrt(_primes(5 * n * math.log(n + 4) / 4))
    q = ps[:, np.newaxis]
    c = None
    dc = None
    for S in range(N):
        vp = on.copy()
        s = np.zeros((n, P))
        for i in range(n):
            x = np.abs(2 * np.mod(q[i] * np.arange(1, P + 1) + rng.random(), 1) - 1)
            if i == 0:
                r = on
                if nu > 0:
                    r = np.sqrt(2 * _gaminv(x, nu / 2))
            else:
                y = _Phinv(c + x * dc)
                s[i:] += ch[i:, i - 1:i] * y
            si = s[i, :]
            c = on.copy()
            ai = az[i] * r - si
            d = on.copy()
            bi = bz[i] * r - si
            c[ai <= -9] = 0
            tl = abs(ai) < 9
            c[tl] = _Phi(ai[tl])
            d[bi <= -9] = 0
            tl = abs(bi) < 9
            d[tl] = _Phi(bi[tl])
            dc = d - c
            vp = vp * dc
        d = (np.mean(vp) - p) / (S + 1)
        p = p + d
        e = (S - 1) * e / (S + 1) + d ** 2
    e = math.sqrt(e)
    return (p, e)