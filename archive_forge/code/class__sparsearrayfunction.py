from __future__ import annotations
import sys
from collections.abc import Hashable, Iterable, Mapping, Sequence
from enum import Enum
from types import ModuleType
from typing import (
import numpy as np
@runtime_checkable
class _sparsearrayfunction(_arrayfunction[_ShapeType_co, _DType_co], Protocol[_ShapeType_co, _DType_co]):
    """
    Sparse duck array supporting NEP 18.

    Corresponds to np.ndarray.
    """

    def todense(self) -> np.ndarray[Any, _DType_co]:
        ...