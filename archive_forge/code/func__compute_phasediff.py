import itertools
import warnings
import numpy as np
from scipy.fft import fftn, ifftn, fftfreq
from scipy import ndimage as ndi
from ._masked_phase_cross_correlation import _masked_phase_cross_correlation
def _compute_phasediff(cross_correlation_max):
    """
    Compute global phase difference between the two images (should be
        zero if images are non-negative).

    Parameters
    ----------
    cross_correlation_max : complex
        The complex value of the cross correlation at its maximum point.
    """
    return np.arctan2(cross_correlation_max.imag, cross_correlation_max.real)