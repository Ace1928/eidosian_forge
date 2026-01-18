from __future__ import annotations
import os
import pathlib
import warnings
import soundfile as sf
import audioread
import numpy as np
import scipy.signal
import soxr
import lazy_loader as lazy
from numba import jit, stencil, guvectorize
from .fft import get_fftlib
from .convert import frames_to_samples, time_to_samples
from .._cache import cache
from .. import util
from ..util.exceptions import ParameterError
from ..util.decorators import deprecated
from ..util.deprecation import Deprecated, rename_kw
from .._typing import _FloatLike_co, _IntLike_co, _SequenceLike
from typing import Any, BinaryIO, Callable, Generator, Optional, Tuple, Union, List
from numpy.typing import DTypeLike, ArrayLike
@deprecated(version='0.10.0', version_removed='1.0')
def __audioread_load(path, offset, duration, dtype: DTypeLike):
    """Load an audio buffer using audioread.

    This loads one block at a time, and then concatenates the results.
    """
    buf = []
    if isinstance(path, tuple(audioread.available_backends())):
        reader = path
    else:
        reader = audioread.audio_open(path)
    with reader as input_file:
        sr_native = input_file.samplerate
        n_channels = input_file.channels
        s_start = int(sr_native * offset) * n_channels
        if duration is None:
            s_end = np.inf
        else:
            s_end = s_start + int(sr_native * duration) * n_channels
        n = 0
        for frame in input_file:
            frame = util.buf_to_float(frame, dtype=dtype)
            n_prev = n
            n = n + len(frame)
            if n < s_start:
                continue
            if s_end < n_prev:
                break
            if s_end < n:
                frame = frame[:int(s_end - n_prev)]
            if n_prev <= s_start <= n:
                frame = frame[s_start - n_prev:]
            buf.append(frame)
    if buf:
        y = np.concatenate(buf)
        if n_channels > 1:
            y = y.reshape((-1, n_channels)).T
    else:
        y = np.empty(0, dtype=dtype)
    return (y, sr_native)