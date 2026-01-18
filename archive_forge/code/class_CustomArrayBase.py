from __future__ import annotations
import copy
import warnings
from abc import abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Generic, cast, overload
import numpy as np
import pytest
from xarray.core.indexing import ExplicitlyIndexed
from xarray.namedarray._typing import (
from xarray.namedarray.core import NamedArray, from_array
class CustomArrayBase(Generic[_ShapeType_co, _DType_co]):

    def __init__(self, array: duckarray[Any, _DType_co]) -> None:
        self.array: duckarray[Any, _DType_co] = array

    @property
    def dtype(self) -> _DType_co:
        return self.array.dtype

    @property
    def shape(self) -> _Shape:
        return self.array.shape