from __future__ import annotations
from collections import abc
import numbers
import operator
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
import pandas._libs.sparse as splib
from pandas._libs.sparse import (
from pandas._libs.tslibs import NaT
from pandas.compat.numpy import function as nv
from pandas.errors import PerformanceWarning
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import (
from pandas.core.dtypes.astype import astype_array
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import (
from pandas.core import arraylike
import pandas.core.algorithms as algos
from pandas.core.arraylike import OpsMixin
from pandas.core.arrays import ExtensionArray
from pandas.core.base import PandasObject
import pandas.core.common as com
from pandas.core.construction import (
from pandas.core.indexers import (
from pandas.core.nanops import check_below_min_count
from pandas.io.formats import printing
def _take_with_fill(self, indices, fill_value=None) -> np.ndarray:
    if fill_value is None:
        fill_value = self.dtype.na_value
    if indices.min() < -1:
        raise ValueError("Invalid value in 'indices'. Must be between -1 and the length of the array.")
    if indices.max() >= len(self):
        raise IndexError("out of bounds value in 'indices'.")
    if len(self) == 0:
        if (indices == -1).all():
            dtype = np.result_type(self.sp_values, type(fill_value))
            taken = np.empty_like(indices, dtype=dtype)
            taken.fill(fill_value)
            return taken
        else:
            raise IndexError('cannot do a non-empty take from an empty axes.')
    sp_indexer = self.sp_index.lookup_array(indices)
    new_fill_indices = indices == -1
    old_fill_indices = (sp_indexer == -1) & ~new_fill_indices
    if self.sp_index.npoints == 0 and old_fill_indices.all():
        taken = np.full(sp_indexer.shape, fill_value=self.fill_value, dtype=self.dtype.subtype)
    elif self.sp_index.npoints == 0:
        _dtype = np.result_type(self.dtype.subtype, type(fill_value))
        taken = np.full(sp_indexer.shape, fill_value=fill_value, dtype=_dtype)
        taken[old_fill_indices] = self.fill_value
    else:
        taken = self.sp_values.take(sp_indexer)
        m0 = sp_indexer[old_fill_indices] < 0
        m1 = sp_indexer[new_fill_indices] < 0
        result_type = taken.dtype
        if m0.any():
            result_type = np.result_type(result_type, type(self.fill_value))
            taken = taken.astype(result_type)
            taken[old_fill_indices] = self.fill_value
        if m1.any():
            result_type = np.result_type(result_type, type(fill_value))
            taken = taken.astype(result_type)
            taken[new_fill_indices] = fill_value
    return taken