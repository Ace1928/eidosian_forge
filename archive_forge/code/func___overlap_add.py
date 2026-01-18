import warnings
import numpy as np
import scipy
import scipy.ndimage
import scipy.signal
import scipy.interpolate
from numba import jit
from . import convert
from .fft import get_fftlib
from .audio import resample
from .._cache import cache
from .. import util
from ..util.exceptions import ParameterError
from ..filters import get_window, semitone_filterbank
from ..filters import window_sumsquare
from numpy.typing import DTypeLike
from typing import Any, Callable, Optional, Tuple, List, Union, overload
from typing_extensions import Literal
from .._typing import _WindowSpec, _PadMode, _PadModeSTFT
@jit(nopython=True, cache=True)
def __overlap_add(y, ytmp, hop_length):
    n_fft = ytmp.shape[-2]
    N = n_fft
    for frame in range(ytmp.shape[-1]):
        sample = frame * hop_length
        if N > y.shape[-1] - sample:
            N = y.shape[-1] - sample
        y[..., sample:sample + N] += ytmp[..., :N, frame]