import warnings
import numpy as np
from scipy.fftpack import fft
from scipy.optimize import brentq
from scipy.signal import convolve, convolve2d
from scipy.signal.windows import gaussian
from scipy.sparse import coo_matrix
from scipy.special import ive  # pylint: disable=no-name-in-module
from ..utils import _cov, _dot, _stack, conditional_jit
def _dct1d(x):
    """Discrete Cosine Transform in 1 Dimension.

    Parameters
    ----------
    x : numpy array
        1 dimensional array of values for which the
        DCT is desired

    Returns
    -------
    output : DTC transformed values
    """
    x_len = len(x)
    even_increasing = np.arange(0, x_len, 2)
    odd_decreasing = np.arange(x_len - 1, 0, -2)
    x = np.concatenate((x[even_increasing], x[odd_decreasing]))
    w_1k = np.r_[1, 2 * np.exp(-(0 + 1j) * np.arange(1, x_len) * np.pi / (2 * x_len))]
    output = np.real(w_1k * fft(x))
    return output