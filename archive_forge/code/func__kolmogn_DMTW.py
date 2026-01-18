import numpy as np
import scipy.special
import scipy.special._ufuncs as scu
from scipy._lib._finite_differences import _derivative
def _kolmogn_DMTW(n, d, cdf=True):
    """Computes the Kolmogorov CDF:  Pr(D_n <= d) using the MTW approach to
    the Durbin matrix algorithm.

    Durbin (1968); Marsaglia, Tsang, Wang (2003). [1], [3].
    """
    if d >= 1.0:
        return _select_and_clip_prob(1.0, 0.0, cdf)
    nd = n * d
    if nd <= 0.5:
        return _select_and_clip_prob(0.0, 1.0, cdf)
    k = int(np.ceil(nd))
    h = k - nd
    m = 2 * k - 1
    H = np.zeros([m, m])
    intm = np.arange(1, m + 1)
    v = 1.0 - h ** intm
    w = np.empty(m)
    fac = 1.0
    for j in intm:
        w[j - 1] = fac
        fac /= j
        v[j - 1] *= fac
    tt = max(2 * h - 1.0, 0) ** m - 2 * h ** m
    v[-1] = (1.0 + tt) * fac
    for i in range(1, m):
        H[i - 1:, i] = w[:m - i + 1]
    H[:, 0] = v
    H[-1, :] = np.flip(v, axis=0)
    Hpwr = np.eye(np.shape(H)[0])
    nn = n
    expnt = 0
    Hexpnt = 0
    while nn > 0:
        if nn % 2:
            Hpwr = np.matmul(Hpwr, H)
            expnt += Hexpnt
        H = np.matmul(H, H)
        Hexpnt *= 2
        if np.abs(H[k - 1, k - 1]) > _EP128:
            H /= _EP128
            Hexpnt += _E128
        nn = nn // 2
    p = Hpwr[k - 1, k - 1]
    for i in range(1, n + 1):
        p = i * p / n
        if np.abs(p) < _EM128:
            p *= _EP128
            expnt -= _E128
    if expnt != 0:
        p = np.ldexp(p, expnt)
    return _select_and_clip_prob(p, 1.0 - p, cdf)