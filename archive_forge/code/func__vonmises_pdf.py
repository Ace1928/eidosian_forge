import warnings
import numpy as np
from scipy.fftpack import fft
from scipy.optimize import brentq
from scipy.signal import convolve, convolve2d
from scipy.signal.windows import gaussian
from scipy.sparse import coo_matrix
from scipy.special import ive  # pylint: disable=no-name-in-module
from ..utils import _cov, _dot, _stack, conditional_jit
def _vonmises_pdf(x, mu, kappa):
    """Calculate vonmises_pdf."""
    if kappa <= 0:
        raise ValueError("Argument 'kappa' must be positive.")
    pdf = 1 / (2 * np.pi * ive(0, kappa)) * np.exp(np.cos(x - mu) - 1) ** kappa
    return pdf