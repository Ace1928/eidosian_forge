import warnings
import numpy as np
from scipy.fftpack import fft
from scipy.optimize import brentq
from scipy.signal import convolve, convolve2d
from scipy.signal.windows import gaussian
from scipy.sparse import coo_matrix
from scipy.special import ive  # pylint: disable=no-name-in-module
from ..utils import _cov, _dot, _stack, conditional_jit
def _bw_scott(x, x_std=None, **kwargs):
    """Scott's Rule."""
    if x_std is None:
        x_std = np.std(x)
    bw = 1.06 * x_std * len(x) ** (-0.2)
    return bw