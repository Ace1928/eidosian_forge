import warnings
import numpy as np
from scipy.fftpack import fft
from scipy.optimize import brentq
from scipy.signal import convolve, convolve2d
from scipy.signal.windows import gaussian
from scipy.sparse import coo_matrix
from scipy.special import ive  # pylint: disable=no-name-in-module
from ..utils import _cov, _dot, _stack, conditional_jit
def _a1inv(x):
    """Compute inverse function.

    Inverse function of the ratio of the first and
    zeroth order Bessel functions of the first kind.

    Returns the value k, such that a1inv(x) = k, i.e. a1(k) = x.
    """
    if 0 <= x < 0.53:
        return 2 * x + x ** 3 + 5 * x ** 5 / 6
    elif x < 0.85:
        return -0.4 + 1.39 * x + 0.43 / (1 - x)
    else:
        return 1 / (x ** 3 - 4 * x ** 2 + 3 * x)