import warnings
import numpy as np
from scipy.fftpack import fft
from scipy.optimize import brentq
from scipy.signal import convolve, convolve2d
from scipy.signal.windows import gaussian
from scipy.sparse import coo_matrix
from scipy.special import ive  # pylint: disable=no-name-in-module
from ..utils import _cov, _dot, _stack, conditional_jit
def _root(function, N, args, x):
    found = False
    N = max(min(1050, N), 50)
    tol = 1e-11 + 0.01 * (N - 50) / 1000
    while not found:
        try:
            bw, res = brentq(function, 0, 0.01, args=args, full_output=True, disp=False)
            found = res.converged
        except ValueError:
            bw = 0
            tol *= 2.0
            found = False
        if bw <= 0 or tol >= 1:
            bw = (_bw_silverman(x) / np.ptp(x)) ** 2
            return bw
    return bw