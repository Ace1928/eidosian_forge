from __future__ import annotations
import scipy.ndimage
import scipy.sparse
import numpy as np
import numba
from numpy.lib.stride_tricks import as_strided
from .._cache import cache
from .exceptions import ParameterError
from .deprecation import Deprecated
from numpy.typing import ArrayLike, DTypeLike
from typing import (
from typing_extensions import Literal
from .._typing import _SequenceLike, _FloatLike_co, _ComplexLike_co
@numba.vectorize(['complex64(float32)', 'complex128(float64)'], nopython=True, cache=True, identity=1)
def _phasor_angles(x) -> np.complex_:
    return np.cos(x) + 1j * np.sin(x)