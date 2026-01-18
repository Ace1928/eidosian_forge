from __future__ import annotations
import operator
import numpy as np
from xarray.core import dtypes, duck_array_ops
class IncludeNumpySameMethods:
    __slots__ = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        inject_numpy_same(cls)