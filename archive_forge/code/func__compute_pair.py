import numpy as np
from scipy import special
from scipy.optimize import OptimizeResult
from scipy.optimize._zeros_py import (  # noqa: F401
def _compute_pair(k, h0):
    h = h0 / 2 ** k
    max = _N_BASE_STEPS * 2 ** k
    j = np.arange(max + 1) if k == 0 else np.arange(1, max + 1, 2)
    jh = j * h
    pi_2 = np.pi / 2
    u1 = pi_2 * np.cosh(jh)
    u2 = pi_2 * np.sinh(jh)
    wj = u1 / np.cosh(u2) ** 2
    xjc = 1 / (np.exp(u2) * np.cosh(u2))
    wj[0] = wj[0] / 2 if k == 0 else wj[0]
    return (xjc, wj)