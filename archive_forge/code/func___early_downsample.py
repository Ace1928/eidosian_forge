import warnings
import numpy as np
from numba import jit
from . import audio
from .intervals import interval_frequencies
from .fft import get_fftlib
from .convert import cqt_frequencies, note_to_hz
from .spectrum import stft, istft
from .pitch import estimate_tuning
from .._cache import cache
from .. import filters
from .. import util
from ..util.exceptions import ParameterError
from numpy.typing import DTypeLike
from typing import Optional, Union, Collection, List
from .._typing import _WindowSpec, _PadMode, _FloatLike_co, _ensure_not_reachable
def __early_downsample(y, sr, hop_length, res_type, n_octaves, nyquist, filter_cutoff, scale):
    """Perform early downsampling on an audio signal, if it applies."""
    downsample_count = __early_downsample_count(nyquist, filter_cutoff, hop_length, n_octaves)
    if downsample_count > 0:
        downsample_factor = 2 ** downsample_count
        hop_length //= downsample_factor
        if y.shape[-1] < downsample_factor:
            raise ParameterError(f'Input signal length={len(y):d} is too short for {n_octaves:d}-octave CQT')
        new_sr = sr / float(downsample_factor)
        y = audio.resample(y, orig_sr=downsample_factor, target_sr=1, res_type=res_type, scale=True)
        if not scale:
            y *= np.sqrt(downsample_factor)
        sr = new_sr
    return (y, sr, hop_length)