import warnings
import numpy as np
import scipy
import numba
from .spectrum import _spectrogram
from . import convert
from .._cache import cache
from .. import util
from .. import sequence
from ..util.exceptions import ParameterError
from numpy.typing import ArrayLike
from typing import Any, Callable, Optional, Tuple, Union
from .._typing import _WindowSpec, _PadMode, _PadModeSTFT
@numba.stencil
def _pi_stencil(x: np.ndarray) -> np.ndarray:
    """Stencil to compute local parabolic interpolation"""
    a = x[1] + x[-1] - 2 * x[0]
    b = (x[1] - x[-1]) / 2
    if np.abs(b) >= np.abs(a):
        return 0
    return -b / a