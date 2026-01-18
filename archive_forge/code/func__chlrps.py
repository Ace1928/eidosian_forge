import math
import numpy as np
from scipy import special
from scipy.stats._qmc import primes_from_2_to
def _chlrps(R, a, b):
    """
    Computes permuted and scaled lower Cholesky factor c for R which may be
    singular, also permuting and scaling integration limit vectors a and b.
    """
    ep = 1e-10
    eps = np.finfo(R.dtype).eps
    n = len(R)
    c = R.copy()
    ap = a.copy()
    bp = b.copy()
    d = np.sqrt(np.maximum(np.diag(c), 0))
    for i in range(n):
        if d[i] > 0:
            c[:, i] /= d[i]
            c[i, :] /= d[i]
            ap[i] /= d[i]
            bp[i] /= d[i]
    y = np.zeros((n, 1))
    sqtp = math.sqrt(2 * math.pi)
    for k in range(n):
        im = k
        ckk = 0
        dem = 1
        s = 0
        for i in range(k, n):
            if c[i, i] > eps:
                cii = math.sqrt(max(c[i, i], 0))
                if i > 0:
                    s = c[i, :k] @ y[:k]
                ai = (ap[i] - s) / cii
                bi = (bp[i] - s) / cii
                de = _Phi(bi) - _Phi(ai)
                if de <= dem:
                    ckk = cii
                    dem = de
                    am = ai
                    bm = bi
                    im = i
        if im > k:
            ap[[im, k]] = ap[[k, im]]
            bp[[im, k]] = bp[[k, im]]
            c[im, im] = c[k, k]
            t = c[im, :k].copy()
            c[im, :k] = c[k, :k]
            c[k, :k] = t
            t = c[im + 1:, im].copy()
            c[im + 1:, im] = c[im + 1:, k]
            c[im + 1:, k] = t
            t = c[k + 1:im, k].copy()
            c[k + 1:im, k] = c[im, k + 1:im].T
            c[im, k + 1:im] = t.T
        if ckk > ep * (k + 1):
            c[k, k] = ckk
            c[k, k + 1:] = 0
            for i in range(k + 1, n):
                c[i, k] = c[i, k] / ckk
                c[i, k + 1:i + 1] = c[i, k + 1:i + 1] - c[i, k] * c[k + 1:i + 1, k].T
            if abs(dem) > ep:
                y[k] = (np.exp(-am ** 2 / 2) - np.exp(-bm ** 2 / 2)) / (sqtp * dem)
            else:
                y[k] = (am + bm) / 2
                if am < -10:
                    y[k] = bm
                elif bm > 10:
                    y[k] = am
            c[k, :k + 1] /= ckk
            ap[k] /= ckk
            bp[k] /= ckk
        else:
            c[k:, k] = 0
            y[k] = (ap[k] + bp[k]) / 2
        pass
    return (c, ap, bp)