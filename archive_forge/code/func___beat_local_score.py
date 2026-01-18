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
def __beat_local_score(onset_envelope, period):
    """Construct the local score for an onset envlope and given period"""
    window = np.exp(-0.5 * (np.arange(-period, period + 1) * 32.0 / period) ** 2)
    return scipy.signal.convolve(__normalize_onsets(onset_envelope), window, 'same')