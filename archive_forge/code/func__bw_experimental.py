import warnings
import numpy as np
from scipy.fftpack import fft
from scipy.optimize import brentq
from scipy.signal import convolve, convolve2d
from scipy.signal.windows import gaussian
from scipy.sparse import coo_matrix
from scipy.special import ive  # pylint: disable=no-name-in-module
from ..utils import _cov, _dot, _stack, conditional_jit
def _bw_experimental(x, grid_counts=None, x_std=None, x_range=None):
    """Experimental bandwidth estimator."""
    bw_silverman = _bw_silverman(x, x_std=x_std)
    bw_isj = _bw_isj(x, grid_counts=grid_counts, x_range=x_range)
    return 0.5 * (bw_silverman + bw_isj)