from statsmodels.compat.python import lrange
import math
import scipy.stats
import numpy as np
from scipy.optimize import fminbound
def _interpolate_v(p, r, v):
    """
    interpolates v based on the values in the A table for the
    scalar value of r and th
    """
    v0, v1, v2 = _select_vs(v, p)
    y0_sq = (_func(A[p, v0], p, r, v0) + 1.0) ** 2.0
    y1_sq = (_func(A[p, v1], p, r, v1) + 1.0) ** 2.0
    y2_sq = (_func(A[p, v2], p, r, v2) + 1.0) ** 2.0
    if v2 > 1e+38:
        v2 = 1e+38
    v_, v0_, v1_, v2_ = (1.0 / v, 1.0 / v0, 1.0 / v1, 1.0 / v2)
    d2 = 2.0 * ((y2_sq - y1_sq) / (v2_ - v1_) - (y0_sq - y1_sq) / (v0_ - v1_)) / (v2_ - v0_)
    if v2_ + v0_ >= v1_ + v1_:
        d1 = (y2_sq - y1_sq) / (v2_ - v1_) - 0.5 * d2 * (v2_ - v1_)
    else:
        d1 = (y1_sq - y0_sq) / (v1_ - v0_) + 0.5 * d2 * (v1_ - v0_)
    d0 = y1_sq
    y = math.sqrt(d2 / 2.0 * (v_ - v1_) ** 2.0 + d1 * (v_ - v1_) + d0)
    return y