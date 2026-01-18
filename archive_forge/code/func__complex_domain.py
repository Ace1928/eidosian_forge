from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.ndimage.filters import maximum_filter, minimum_filter
from ..audio.signal import smooth as smooth_signal
from ..processors import (BufferProcessor, OnlineProcessor, ParallelProcessor,
from ..utils import combine_events
def _complex_domain(spectrogram):
    """
    Helper method used by complex_domain() & rectified_complex_domain().

    Parameters
    ----------
    spectrogram : :class:`Spectrogram` instance
        :class:`Spectrogram` instance.

    Returns
    -------
    numpy array
        Complex domain onset detection function.

    Notes
    -----
    We use the simple implementation presented in [1]_.

    References
    ----------
    .. [1] Simon Dixon,
           "Onset Detection Revisited",
           Proceedings of the 9th International Conference on Digital Audio
           Effects (DAFx), 2006.

    """
    phase = spectrogram.stft.phase()
    if np.shape(phase) != np.shape(spectrogram):
        raise ValueError('spectrogram and phase must be of same shape')
    cd_target = np.zeros_like(phase)
    cd_target[1:] = 2 * phase[1:] - phase[:-1]
    cd_target = spectrogram * np.exp(1j * cd_target)
    cd = spectrogram * np.exp(1j * phase)
    cd[1:] -= cd_target[:-1]
    return np.asarray(cd)