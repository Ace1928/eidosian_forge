from __future__ import annotations
import sys
from collections.abc import Hashable, Iterable, Mapping, Sequence
from enum import Enum
from types import ModuleType
from typing import (
import numpy as np
@runtime_checkable
class _arrayfunction(_array[_ShapeType_co, _DType_co], Protocol[_ShapeType_co, _DType_co]):
    """
    Duck array supporting NEP 18.

    Corresponds to np.ndarray.
    """

    @overload
    def __getitem__(self, key: _arrayfunction[Any, Any] | tuple[_arrayfunction[Any, Any], ...], /) -> _arrayfunction[Any, _DType_co]:
        ...

    @overload
    def __getitem__(self, key: _IndexKeyLike, /) -> Any:
        ...

    def __getitem__(self, key: _IndexKeyLike | _arrayfunction[Any, Any] | tuple[_arrayfunction[Any, Any], ...], /) -> _arrayfunction[Any, _DType_co] | Any:
        ...

    @overload
    def __array__(self, dtype: None=..., /) -> np.ndarray[Any, _DType_co]:
        ...

    @overload
    def __array__(self, dtype: _DType, /) -> np.ndarray[Any, _DType]:
        ...

    def __array__(self, dtype: _DType | None=..., /) -> np.ndarray[Any, _DType] | np.ndarray[Any, _DType_co]:
        ...

    def __array_ufunc__(self, ufunc: Any, method: Any, *inputs: Any, **kwargs: Any) -> Any:
        ...

    def __array_function__(self, func: Callable[..., Any], types: Iterable[type], args: Iterable[Any], kwargs: Mapping[str, Any]) -> Any:
        ...

    @property
    def imag(self) -> _arrayfunction[_ShapeType_co, Any]:
        ...

    @property
    def real(self) -> _arrayfunction[_ShapeType_co, Any]:
        ...