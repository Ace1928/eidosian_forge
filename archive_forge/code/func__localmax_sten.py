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
@numba.stencil
def _localmax_sten(x):
    """Numba stencil for local maxima computation"""
    return (x[0] > x[-1]) & (x[0] >= x[1])