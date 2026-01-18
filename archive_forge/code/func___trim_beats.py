import numpy as np
import scipy
import scipy.stats
from ._cache import cache
from . import core
from . import onset
from . import util
from .feature import tempogram, fourier_tempogram
from .feature import tempo as _tempo
from .util.exceptions import ParameterError
from .util.decorators import moved
from typing import Any, Callable, Optional, Tuple
def __trim_beats(localscore: np.ndarray, beats: np.ndarray, trim: bool) -> np.ndarray:
    """Remove spurious leading and trailing beats"""
    smooth_boe = scipy.signal.convolve(localscore[beats], scipy.signal.hann(5), 'same')
    if trim:
        threshold = 0.5 * (smooth_boe ** 2).mean() ** 0.5
    else:
        threshold = 0.0
    valid = np.argwhere(smooth_boe > threshold)
    return beats[valid.min():valid.max()]