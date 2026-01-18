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
def _numba_apply(self, func: Callable[..., Any], engine_kwargs: dict[str, bool] | None=None, **func_kwargs):
    window_indexer = self._get_window_indexer()
    min_periods = self.min_periods if self.min_periods is not None else window_indexer.window_size
    obj = self._create_data(self._selected_obj)
    if self.axis == 1:
        obj = obj.T
    values = self._prep_values(obj.to_numpy())
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    start, end = window_indexer.get_window_bounds(num_values=len(values), min_periods=min_periods, center=self.center, closed=self.closed, step=self.step)
    self._check_window_bounds(start, end, len(values))
    dtype_mapping = executor.float_dtype_mapping
    aggregator = executor.generate_shared_aggregator(func, dtype_mapping, is_grouped_kernel=False, **get_jit_arguments(engine_kwargs))
    result = aggregator(values.T, start=start, end=end, min_periods=min_periods, **func_kwargs).T
    result = result.T if self.axis == 1 else result
    index = self._slice_axis_for_step(obj.index, result)
    if obj.ndim == 1:
        result = result.squeeze()
        out = obj._constructor(result, index=index, name=obj.name)
        return out
    else:
        columns = self._slice_axis_for_step(obj.columns, result.T)
        out = obj._constructor(result, index=index, columns=columns)
        return self._resolve_output(out, obj)