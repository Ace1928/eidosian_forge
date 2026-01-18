from __future__ import annotations
from collections.abc import (
from typing import (
import numpy as np
from pandas._libs import algos as libalgos
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import BaseMaskedDtype
class SelectN:

    def __init__(self, obj, n: int, keep: str) -> None:
        self.obj = obj
        self.n = n
        self.keep = keep
        if self.keep not in ('first', 'last', 'all'):
            raise ValueError('keep must be either "first", "last" or "all"')

    def compute(self, method: str) -> DataFrame | Series:
        raise NotImplementedError

    @final
    def nlargest(self):
        return self.compute('nlargest')

    @final
    def nsmallest(self):
        return self.compute('nsmallest')

    @final
    @staticmethod
    def is_valid_dtype_n_method(dtype: DtypeObj) -> bool:
        """
        Helper function to determine if dtype is valid for
        nsmallest/nlargest methods
        """
        if is_numeric_dtype(dtype):
            return not is_complex_dtype(dtype)
        return needs_i8_conversion(dtype)