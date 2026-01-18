from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.ndimage.filters import maximum_filter, minimum_filter
from ..audio.signal import smooth as smooth_signal
from ..processors import (BufferProcessor, OnlineProcessor, ParallelProcessor,
from ..utils import combine_events
def _phase_deviation(phase):
    """
    Helper function used by phase_deviation() & weighted_phase_deviation().

    Parameters
    ----------
    phase : numpy array
        Phase of the STFT.

    Returns
    -------
    numpy array
        Phase deviation.

    """
    pd = np.zeros_like(phase)
    pd[2:] = phase[2:] - 2 * phase[1:-1] + phase[:-2]
    return np.asarray(wrap_to_pi(pd))