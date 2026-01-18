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
def __soundfile_load(path, offset, duration, dtype):
    """Load an audio buffer using soundfile."""
    if isinstance(path, sf.SoundFile):
        context = path
    else:
        context = sf.SoundFile(path)
    with context as sf_desc:
        sr_native = sf_desc.samplerate
        if offset:
            sf_desc.seek(int(offset * sr_native))
        if duration is not None:
            frame_duration = int(duration * sr_native)
        else:
            frame_duration = -1
        y = sf_desc.read(frames=frame_duration, dtype=dtype, always_2d=False).T
    return (y, sr_native)