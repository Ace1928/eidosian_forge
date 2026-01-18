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
def __cqt_response(y, n_fft, hop_length, fft_basis, mode, window='ones', phase=True, dtype=None):
    """Compute the filter response with a target STFT hop."""
    D = stft(y, n_fft=n_fft, hop_length=hop_length, window=window, pad_mode=mode, dtype=dtype)
    if not phase:
        D = np.abs(D)
    Dr = D.reshape((-1, D.shape[-2], D.shape[-1]))
    output_flat = np.empty((Dr.shape[0], fft_basis.shape[0], Dr.shape[-1]), dtype=D.dtype)
    for i in range(Dr.shape[0]):
        output_flat[i] = fft_basis.dot(Dr[i])
    shape = list(D.shape)
    shape[-2] = fft_basis.shape[0]
    return output_flat.reshape(shape)