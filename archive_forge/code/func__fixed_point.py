import warnings
import numpy as np
from scipy.fftpack import fft
from scipy.optimize import brentq
from scipy.signal import convolve, convolve2d
from scipy.signal.windows import gaussian
from scipy.sparse import coo_matrix
from scipy.special import ive  # pylint: disable=no-name-in-module
from ..utils import _cov, _dot, _stack, conditional_jit
def _fixed_point(t, N, k_sq, a_sq):
    """Calculate t-zeta*gamma^[l](t).

    Implementation of the function t-zeta*gamma^[l](t) derived from equation (30) in [1].

    References
    ----------
    .. [1] Kernel density estimation via diffusion.
       Z. I. Botev, J. F. Grotowski, and D. P. Kroese.
       Ann. Statist. 38 (2010), no. 5, 2916--2957.
    """
    k_sq = np.asarray(k_sq, dtype=np.float64)
    a_sq = np.asarray(a_sq, dtype=np.float64)
    l = 7
    f = np.sum(np.power(k_sq, l) * a_sq * np.exp(-k_sq * np.pi ** 2 * t))
    f *= 0.5 * np.pi ** (2.0 * l)
    for j in np.arange(l - 1, 2 - 1, -1):
        c1 = (1 + 0.5 ** (j + 0.5)) / 3
        c2 = np.prod(np.arange(1.0, 2 * j + 1, 2, dtype=np.float64))
        c2 /= (np.pi / 2) ** 0.5
        t_j = np.power(c1 * (c2 / (N * f)), 2.0 / (3.0 + 2.0 * j))
        f = np.sum(k_sq ** j * a_sq * np.exp(-k_sq * np.pi ** 2.0 * t_j))
        f *= 0.5 * np.pi ** (2 * j)
    out = t - (2 * N * np.pi ** 0.5 * f) ** (-0.4)
    return out