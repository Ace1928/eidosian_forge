import warnings
import numpy as np
import scipy
import numba
from .spectrum import _spectrogram
from . import convert
from .._cache import cache
from .. import util
from .. import sequence
from ..util.exceptions import ParameterError
from numpy.typing import ArrayLike
from typing import Any, Callable, Optional, Tuple, Union
from .._typing import _WindowSpec, _PadMode, _PadModeSTFT
def _cumulative_mean_normalized_difference(y_frames: np.ndarray, frame_length: int, win_length: int, min_period: int, max_period: int) -> np.ndarray:
    """Cumulative mean normalized difference function (equation 8 in [#]_)

    .. [#] De Cheveigné, Alain, and Hideki Kawahara.
        "YIN, a fundamental frequency estimator for speech and music."
        The Journal of the Acoustical Society of America 111.4 (2002): 1917-1930.

    Parameters
    ----------
    y_frames : np.ndarray [shape=(frame_length, n_frames)]
        framed audio time series.
    frame_length : int > 0 [scalar]
        length of the frames in samples.
    win_length : int > 0 [scalar]
        length of the window for calculating autocorrelation in samples.
    min_period : int > 0 [scalar]
        minimum period.
    max_period : int > 0 [scalar]
        maximum period.

    Returns
    -------
    yin_frames : np.ndarray [shape=(max_period-min_period+1,n_frames)]
        Cumulative mean normalized difference function for each frame.
    """
    a = np.fft.rfft(y_frames, frame_length, axis=-2)
    b = np.fft.rfft(y_frames[..., win_length:0:-1, :], frame_length, axis=-2)
    acf_frames = np.fft.irfft(a * b, frame_length, axis=-2)[..., win_length:, :]
    acf_frames[np.abs(acf_frames) < 1e-06] = 0
    energy_frames = np.cumsum(y_frames ** 2, axis=-2)
    energy_frames = energy_frames[..., win_length:, :] - energy_frames[..., :-win_length, :]
    energy_frames[np.abs(energy_frames) < 1e-06] = 0
    yin_frames = energy_frames[..., :1, :] + energy_frames - 2 * acf_frames
    yin_numerator = yin_frames[..., min_period:max_period + 1, :]
    tau_range = util.expand_to(np.arange(1, max_period + 1), ndim=yin_frames.ndim, axes=-2)
    cumulative_mean = np.cumsum(yin_frames[..., 1:max_period + 1, :], axis=-2) / tau_range
    yin_denominator = cumulative_mean[..., min_period - 1:max_period, :]
    yin_frames: np.ndarray = yin_numerator / (yin_denominator + util.tiny(yin_denominator))
    return yin_frames