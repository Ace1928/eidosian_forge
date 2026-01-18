import warnings
import numpy as np
from scipy.fftpack import fft
from scipy.optimize import brentq
from scipy.signal import convolve, convolve2d
from scipy.signal.windows import gaussian
from scipy.sparse import coo_matrix
from scipy.special import ive  # pylint: disable=no-name-in-module
from ..utils import _cov, _dot, _stack, conditional_jit
def _bw_silverman(x, x_std=None, **kwargs):
    """Silverman's Rule."""
    if x_std is None:
        x_std = np.std(x)
    q75, q25 = np.percentile(x, [75, 25])
    x_iqr = q75 - q25
    a = min(x_std, x_iqr / 1.34)
    bw = 0.9 * a * len(x) ** (-0.2)
    return bw