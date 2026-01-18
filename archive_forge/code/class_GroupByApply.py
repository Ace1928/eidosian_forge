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
class GroupByApply(Apply):
    obj: GroupBy | Resampler | BaseWindow

    def __init__(self, obj: GroupBy[NDFrameT], func: AggFuncType, *, args, kwargs) -> None:
        kwargs = kwargs.copy()
        self.axis = obj.obj._get_axis_number(kwargs.get('axis', 0))
        super().__init__(obj, func, raw=False, result_type=None, args=args, kwargs=kwargs)

    def apply(self):
        raise NotImplementedError

    def transform(self):
        raise NotImplementedError

    def agg_or_apply_list_like(self, op_name: Literal['agg', 'apply']) -> DataFrame | Series:
        obj = self.obj
        kwargs = self.kwargs
        if op_name == 'apply':
            kwargs = {**kwargs, 'by_row': False}
        if getattr(obj, 'axis', 0) == 1:
            raise NotImplementedError('axis other than 0 is not supported')
        if obj._selected_obj.ndim == 1:
            selected_obj = obj._selected_obj
        else:
            selected_obj = obj._obj_with_exclusions
        with com.temp_setattr(obj, 'as_index', True, condition=hasattr(obj, 'as_index')):
            keys, results = self.compute_list_like(op_name, selected_obj, kwargs)
        result = self.wrap_results_list_like(keys, results)
        return result

    def agg_or_apply_dict_like(self, op_name: Literal['agg', 'apply']) -> DataFrame | Series:
        from pandas.core.groupby.generic import DataFrameGroupBy, SeriesGroupBy
        assert op_name in ['agg', 'apply']
        obj = self.obj
        kwargs = {}
        if op_name == 'apply':
            by_row = '_compat' if self.by_row else False
            kwargs.update({'by_row': by_row})
        if getattr(obj, 'axis', 0) == 1:
            raise NotImplementedError('axis other than 0 is not supported')
        selected_obj = obj._selected_obj
        selection = obj._selection
        is_groupby = isinstance(obj, (DataFrameGroupBy, SeriesGroupBy))
        if is_groupby:
            engine = self.kwargs.get('engine', None)
            engine_kwargs = self.kwargs.get('engine_kwargs', None)
            kwargs.update({'engine': engine, 'engine_kwargs': engine_kwargs})
        with com.temp_setattr(obj, 'as_index', True, condition=hasattr(obj, 'as_index')):
            result_index, result_data = self.compute_dict_like(op_name, selected_obj, selection, kwargs)
        result = self.wrap_results_dict_like(selected_obj, result_index, result_data)
        return result