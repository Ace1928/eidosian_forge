from statsmodels.compat.python import lrange
import math
import scipy.stats
import numpy as np
from scipy.optimize import fminbound
def _qsturng(p, r, v):
    """scalar version of qsturng"""
    global A, p_keys, v_keys
    if p < 0.1 or p > 0.999:
        raise ValueError('p must be between .1 and .999')
    if p < 0.9:
        if v < 2:
            raise ValueError('v must be > 2 when p < .9')
    elif v < 1:
        raise ValueError('v must be > 1 when p >= .9')
    p = float(p)
    if isinstance(v, np.ndarray):
        v = v.item()
    if (p, v) in A:
        y = _func(A[p, v], p, r, v) + 1.0
    elif p not in p_keys and v not in v_keys + ([], [1])[p >= 0.9]:
        v0, v1, v2 = _select_vs(v, p)
        p0, p1, p2 = _select_ps(p)
        r0_sq = _interpolate_p(p, r, v0) ** 2
        r1_sq = _interpolate_p(p, r, v1) ** 2
        r2_sq = _interpolate_p(p, r, v2) ** 2
        v_, v0_, v1_, v2_ = (1.0 / v, 1.0 / v0, 1.0 / v1, 1.0 / v2)
        d2 = 2.0 * ((r2_sq - r1_sq) / (v2_ - v1_) - (r0_sq - r1_sq) / (v0_ - v1_)) / (v2_ - v0_)
        if v2_ + v0_ >= v1_ + v1_:
            d1 = (r2_sq - r1_sq) / (v2_ - v1_) - 0.5 * d2 * (v2_ - v1_)
        else:
            d1 = (r1_sq - r0_sq) / (v1_ - v0_) + 0.5 * d2 * (v1_ - v0_)
        d0 = r1_sq
        y = math.sqrt(d2 / 2.0 * (v_ - v1_) ** 2.0 + d1 * (v_ - v1_) + d0)
    elif v not in v_keys + ([], [1])[p >= 0.9]:
        y = _interpolate_v(p, r, v)
    elif p not in p_keys:
        y = _interpolate_p(p, r, v)
    v = min(v, 1e+38)
    return math.sqrt(2) * -y * scipy.stats.t.isf((1.0 + p) / 2.0, v)