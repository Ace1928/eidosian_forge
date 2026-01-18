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
@numba.jit(nopython=True, cache=True)
def __shear_dense(X: np.ndarray, *, factor: int=+1, axis: int=-1) -> np.ndarray:
    """Numba-accelerated shear for dense (ndarray) arrays"""
    if axis == 0:
        X = X.T
    X_shear = np.empty_like(X)
    for i in range(X.shape[1]):
        X_shear[:, i] = np.roll(X[:, i], factor * i)
    if axis == 0:
        X_shear = X_shear.T
    return X_shear