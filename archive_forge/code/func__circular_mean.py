import warnings
import numpy as np
from scipy.fftpack import fft
from scipy.optimize import brentq
from scipy.signal import convolve, convolve2d
from scipy.signal.windows import gaussian
from scipy.sparse import coo_matrix
from scipy.special import ive  # pylint: disable=no-name-in-module
from ..utils import _cov, _dot, _stack, conditional_jit
def _circular_mean(x):
    """Compute mean of circular variable measured in radians.

    The result is between -pi and pi.
    """
    sinr = np.sum(np.sin(x))
    cosr = np.sum(np.cos(x))
    mean = np.arctan2(sinr, cosr)
    return mean