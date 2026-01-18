from __future__ import annotations
import math
import typing as ty
from dataclasses import dataclass, replace
import numpy as np
from nibabel.casting import able_int_type
from nibabel.fileslice import strided_scalar
from nibabel.spatialimages import SpatialImage
class CoordinateArray(ty.Protocol):
    ndim: int
    shape: tuple[int, int]

    @ty.overload
    def __array__(self, dtype: None=..., /) -> np.ndarray[ty.Any, np.dtype[ty.Any]]:
        ...

    @ty.overload
    def __array__(self, dtype: _DType, /) -> np.ndarray[ty.Any, _DType]:
        ...