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
@numba.guvectorize(['void(int16[:], bool_[:])', 'void(int32[:], bool_[:])', 'void(int64[:], bool_[:])', 'void(float32[:], bool_[:])', 'void(float64[:], bool_[:])'], '(n)->(n)', cache=True, nopython=True)
def _localmin(x, y):
    """Vectorized wrapper for the localmin stencil"""
    y[:] = _localmin_sten(x)