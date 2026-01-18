from __future__ import annotations
import abc
from collections import defaultdict
import functools
from functools import partial
import inspect
from typing import (
import warnings
import numpy as np
from pandas._config import option_context
from pandas._libs import lib
from pandas._libs.internals import BlockValuesRefs
from pandas._typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.errors import SpecificationError
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import is_nested_object
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core._numba.executor import generate_apply_looper
import pandas.core.common as com
from pandas.core.construction import ensure_wrapped_if_datetimelike
class FrameRowApply(FrameApply):
    axis: AxisInt = 0

    @property
    def series_generator(self) -> Generator[Series, None, None]:
        return (self.obj._ixs(i, axis=1) for i in range(len(self.columns)))

    @staticmethod
    @functools.cache
    def generate_numba_apply_func(func, nogil=True, nopython=True, parallel=False) -> Callable[[npt.NDArray, Index, Index], dict[int, Any]]:
        numba = import_optional_dependency('numba')
        from pandas import Series
        from pandas.core._numba.extensions import maybe_cast_str
        jitted_udf = numba.extending.register_jitable(func)

        @numba.jit(nogil=nogil, nopython=nopython, parallel=parallel)
        def numba_func(values, col_names, df_index):
            results = {}
            for j in range(values.shape[1]):
                ser = Series(values[:, j], index=df_index, name=maybe_cast_str(col_names[j]))
                results[j] = jitted_udf(ser)
            return results
        return numba_func

    def apply_with_numba(self) -> dict[int, Any]:
        nb_func = self.generate_numba_apply_func(cast(Callable, self.func), **self.engine_kwargs)
        from pandas.core._numba.extensions import set_numba_data
        index = self.obj.index
        if index.dtype == 'string':
            index = index.astype(object)
        columns = self.obj.columns
        if columns.dtype == 'string':
            columns = columns.astype(object)
        with set_numba_data(index) as index, set_numba_data(columns) as columns:
            res = dict(nb_func(self.values, columns, index))
        return res

    @property
    def result_index(self) -> Index:
        return self.columns

    @property
    def result_columns(self) -> Index:
        return self.index

    def wrap_results_for_axis(self, results: ResType, res_index: Index) -> DataFrame | Series:
        """return the results for the rows"""
        if self.result_type == 'reduce':
            res = self.obj._constructor_sliced(results)
            res.index = res_index
            return res
        elif self.result_type is None and all((isinstance(x, dict) for x in results.values())):
            res = self.obj._constructor_sliced(results)
            res.index = res_index
            return res
        try:
            result = self.obj._constructor(data=results)
        except ValueError as err:
            if 'All arrays must be of the same length' in str(err):
                res = self.obj._constructor_sliced(results)
                res.index = res_index
                return res
            else:
                raise
        if not isinstance(results[0], ABCSeries):
            if len(result.index) == len(self.res_columns):
                result.index = self.res_columns
        if len(result.columns) == len(res_index):
            result.columns = res_index
        return result