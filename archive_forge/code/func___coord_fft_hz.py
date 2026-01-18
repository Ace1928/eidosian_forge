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
def __coord_fft_hz(n: int, sr: float=22050, n_fft: Optional[int]=None, **_kwargs: Any) -> np.ndarray:
    """Get the frequencies for FFT bins"""
    if n_fft is None:
        n_fft = 2 * (n - 1)
    basis = core.fft_frequencies(sr=sr, n_fft=n_fft)
    return basis