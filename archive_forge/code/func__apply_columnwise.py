from __future__ import annotations
import copy
from datetime import timedelta
from functools import partial
import inspect
from textwrap import dedent
from typing import (
import numpy as np
from pandas._libs.tslibs import (
import pandas._libs.window.aggregations as window_aggregations
from pandas.compat._optional import import_optional_dependency
from pandas.errors import DataError
from pandas.util._decorators import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import notna
from pandas.core._numba import executor
from pandas.core.algorithms import factorize
from pandas.core.apply import ResamplerWindowApply
from pandas.core.arrays import ExtensionArray
from pandas.core.base import SelectionMixin
import pandas.core.common as com
from pandas.core.indexers.objects import (
from pandas.core.indexes.api import (
from pandas.core.reshape.concat import concat
from pandas.core.util.numba_ import (
from pandas.core.window.common import (
from pandas.core.window.doc import (
from pandas.core.window.numba_ import (
from pandas.core.arrays.datetimelike import dtype_to_unit
def _apply_columnwise(self, homogeneous_func: Callable[..., ArrayLike], name: str, numeric_only: bool=False) -> DataFrame | Series:
    """
        Apply the given function to the DataFrame broken down into homogeneous
        sub-frames.
        """
    self._validate_numeric_only(name, numeric_only)
    if self._selected_obj.ndim == 1:
        return self._apply_series(homogeneous_func, name)
    obj = self._create_data(self._selected_obj, numeric_only)
    if name == 'count':
        obj = notna(obj).astype(int)
        obj._mgr = obj._mgr.consolidate()
    if self.axis == 1:
        obj = obj.T
    taker = []
    res_values = []
    for i, arr in enumerate(obj._iter_column_arrays()):
        try:
            arr = self._prep_values(arr)
        except (TypeError, NotImplementedError) as err:
            raise DataError(f'Cannot aggregate non-numeric type: {arr.dtype}') from err
        res = homogeneous_func(arr)
        res_values.append(res)
        taker.append(i)
    index = self._slice_axis_for_step(obj.index, res_values[0] if len(res_values) > 0 else None)
    df = type(obj)._from_arrays(res_values, index=index, columns=obj.columns.take(taker), verify_integrity=False)
    if self.axis == 1:
        df = df.T
    return self._resolve_output(df, obj)