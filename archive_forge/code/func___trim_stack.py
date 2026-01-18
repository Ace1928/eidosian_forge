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
def __trim_stack(cqt_resp: List[np.ndarray], n_bins: int, dtype: DTypeLike) -> np.ndarray:
    """Trim and stack a collection of CQT responses"""
    max_col = min((c_i.shape[-1] for c_i in cqt_resp))
    shape = list(cqt_resp[0].shape)
    shape[-2] = n_bins
    shape[-1] = max_col
    cqt_out = np.empty(shape, dtype=dtype, order='F')
    end = n_bins
    for c_i in cqt_resp:
        n_oct = c_i.shape[-2]
        if end < n_oct:
            cqt_out[..., :end, :] = c_i[..., -end:, :max_col]
        else:
            cqt_out[..., end - n_oct:end, :] = c_i[..., :max_col]
        end -= n_oct
    return cqt_out