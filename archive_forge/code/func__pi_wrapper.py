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
@numba.guvectorize(['void(float32[:], float32[:])', 'void(float64[:], float64[:])'], '(n)->(n)', cache=True, nopython=True)
def _pi_wrapper(x: np.ndarray, y: np.ndarray) -> None:
    """Vectorized wrapper for the parabolic interpolation stencil"""
    y[:] = _pi_stencil(x)