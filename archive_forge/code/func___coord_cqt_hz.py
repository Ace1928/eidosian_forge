from __future__ import annotations
from itertools import product
import warnings
import numpy as np
import matplotlib.cm as mcm
import matplotlib.axes as mplaxes
import matplotlib.ticker as mplticker
import matplotlib.pyplot as plt
from . import core
from . import util
from .util.deprecation import rename_kw, Deprecated
from .util.exceptions import ParameterError
from typing import TYPE_CHECKING, Any, Collection, Optional, Union, Callable, Dict
from ._typing import _FloatLike_co
def __coord_cqt_hz(n: int, fmin: Optional[_FloatLike_co]=None, bins_per_octave: int=12, sr: float=22050, **_kwargs: Any) -> np.ndarray:
    """Get CQT bin frequencies"""
    if fmin is None:
        fmin = core.note_to_hz('C1')
    fmin = fmin * 2.0 ** (_kwargs.get('tuning', 0.0) / bins_per_octave)
    freqs = core.cqt_frequencies(n, fmin=fmin, bins_per_octave=bins_per_octave)
    if np.any(freqs > 0.5 * sr):
        warnings.warn('Frequency axis exceeds Nyquist. Did you remember to set all spectrogram parameters in specshow?', stacklevel=4)
    return freqs