import warnings
import numpy as np
from scipy.fftpack import fft
from scipy.optimize import brentq
from scipy.signal import convolve, convolve2d
from scipy.signal.windows import gaussian
from scipy.sparse import coo_matrix
from scipy.special import ive  # pylint: disable=no-name-in-module
from ..utils import _cov, _dot, _stack, conditional_jit
def _find_hdi_contours(density, hdi_probs):
    """
    Find contours enclosing regions of highest posterior density.

    Parameters
    ----------
    density : array-like
        A 2D KDE on a grid with cells of equal area.
    hdi_probs : array-like
        An array of highest density interval confidence probabilities.

    Returns
    -------
    contour_levels : array
        The contour levels corresponding to the given HDI probabilities.
    """
    sorted_density = np.sort(density, axis=None)[::-1]
    sm = sorted_density.cumsum()
    sm /= sm[-1]
    contours = np.empty_like(hdi_probs)
    for idx, hdi_prob in enumerate(hdi_probs):
        try:
            contours[idx] = sorted_density[sm <= hdi_prob][-1]
        except IndexError:
            contours[idx] = sorted_density[0]
    return contours