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
class FrameColumnApply(FrameApply):
    axis: AxisInt = 1

    def apply_broadcast(self, target: DataFrame) -> DataFrame:
        result = super().apply_broadcast(target.T)
        return result.T

    @property
    def series_generator(self) -> Generator[Series, None, None]:
        values = self.values
        values = ensure_wrapped_if_datetimelike(values)
        assert len(values) > 0
        ser = self.obj._ixs(0, axis=0)
        mgr = ser._mgr
        is_view = mgr.blocks[0].refs.has_reference()
        if isinstance(ser.dtype, ExtensionDtype):
            obj = self.obj
            for i in range(len(obj)):
                yield obj._ixs(i, axis=0)
        else:
            for arr, name in zip(values, self.index):
                ser._mgr = mgr
                mgr.set_values(arr)
                object.__setattr__(ser, '_name', name)
                if not is_view:
                    mgr.blocks[0].refs = BlockValuesRefs(mgr.blocks[0])
                yield ser

    @staticmethod
    @functools.cache
    def generate_numba_apply_func(func, nogil=True, nopython=True, parallel=False) -> Callable[[npt.NDArray, Index, Index], dict[int, Any]]:
        numba = import_optional_dependency('numba')
        from pandas import Series
        from pandas.core._numba.extensions import maybe_cast_str
        jitted_udf = numba.extending.register_jitable(func)

        @numba.jit(nogil=nogil, nopython=nopython, parallel=parallel)
        def numba_func(values, col_names_index, index):
            results = {}
            for i in range(values.shape[0]):
                ser = Series(values[i].copy(), index=col_names_index, name=maybe_cast_str(index[i]))
                results[i] = jitted_udf(ser)
            return results
        return numba_func

    def apply_with_numba(self) -> dict[int, Any]:
        nb_func = self.generate_numba_apply_func(cast(Callable, self.func), **self.engine_kwargs)
        from pandas.core._numba.extensions import set_numba_data
        with set_numba_data(self.obj.index) as index, set_numba_data(self.columns) as columns:
            res = dict(nb_func(self.values, columns, index))
        return res

    @property
    def result_index(self) -> Index:
        return self.index

    @property
    def result_columns(self) -> Index:
        return self.columns

    def wrap_results_for_axis(self, results: ResType, res_index: Index) -> DataFrame | Series:
        """return the results for the columns"""
        result: DataFrame | Series
        if self.result_type == 'expand':
            result = self.infer_to_same_shape(results, res_index)
        elif not isinstance(results[0], ABCSeries):
            result = self.obj._constructor_sliced(results)
            result.index = res_index
        else:
            result = self.infer_to_same_shape(results, res_index)
        return result

    def infer_to_same_shape(self, results: ResType, res_index: Index) -> DataFrame:
        """infer the results to the same shape as the input object"""
        result = self.obj._constructor(data=results)
        result = result.T
        result.index = res_index
        result = result.infer_objects(copy=False)
        return result