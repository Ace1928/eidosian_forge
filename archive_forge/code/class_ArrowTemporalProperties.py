from __future__ import annotations
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import ABCSeries
from pandas.core.accessor import (
from pandas.core.arrays import (
from pandas.core.arrays.arrow.array import ArrowExtensionArray
from pandas.core.base import (
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.indexes.timedeltas import TimedeltaIndex
@delegate_names(delegate=ArrowExtensionArray, accessors=TimedeltaArray._datetimelike_ops, typ='property', accessor_mapping=lambda x: f'_dt_{x}', raise_on_missing=False)
@delegate_names(delegate=ArrowExtensionArray, accessors=TimedeltaArray._datetimelike_methods, typ='method', accessor_mapping=lambda x: f'_dt_{x}', raise_on_missing=False)
@delegate_names(delegate=ArrowExtensionArray, accessors=DatetimeArray._datetimelike_ops, typ='property', accessor_mapping=lambda x: f'_dt_{x}', raise_on_missing=False)
@delegate_names(delegate=ArrowExtensionArray, accessors=DatetimeArray._datetimelike_methods, typ='method', accessor_mapping=lambda x: f'_dt_{x}', raise_on_missing=False)
class ArrowTemporalProperties(PandasDelegate, PandasObject, NoNewAttributesMixin):

    def __init__(self, data: Series, orig) -> None:
        if not isinstance(data, ABCSeries):
            raise TypeError(f'cannot convert an object of type {type(data)} to a datetimelike index')
        self._parent = data
        self._orig = orig
        self._freeze()

    def _delegate_property_get(self, name: str):
        if not hasattr(self._parent.array, f'_dt_{name}'):
            raise NotImplementedError(f'dt.{name} is not supported for {self._parent.dtype}')
        result = getattr(self._parent.array, f'_dt_{name}')
        if not is_list_like(result):
            return result
        if self._orig is not None:
            index = self._orig.index
        else:
            index = self._parent.index
        result = type(self._parent)(result, index=index, name=self._parent.name).__finalize__(self._parent)
        return result

    def _delegate_method(self, name: str, *args, **kwargs):
        if not hasattr(self._parent.array, f'_dt_{name}'):
            raise NotImplementedError(f'dt.{name} is not supported for {self._parent.dtype}')
        result = getattr(self._parent.array, f'_dt_{name}')(*args, **kwargs)
        if self._orig is not None:
            index = self._orig.index
        else:
            index = self._parent.index
        result = type(self._parent)(result, index=index, name=self._parent.name).__finalize__(self._parent)
        return result

    def to_pytimedelta(self):
        return cast(ArrowExtensionArray, self._parent.array)._dt_to_pytimedelta()

    def to_pydatetime(self):
        warnings.warn(f'The behavior of {type(self).__name__}.to_pydatetime is deprecated, in a future version this will return a Series containing python datetime objects instead of an ndarray. To retain the old behavior, call `np.array` on the result', FutureWarning, stacklevel=find_stack_level())
        return cast(ArrowExtensionArray, self._parent.array)._dt_to_pydatetime()

    def isocalendar(self) -> DataFrame:
        from pandas import DataFrame
        result = cast(ArrowExtensionArray, self._parent.array)._dt_isocalendar()._pa_array.combine_chunks()
        iso_calendar_df = DataFrame({col: type(self._parent.array)(result.field(i)) for i, col in enumerate(['year', 'week', 'day'])})
        return iso_calendar_df

    @property
    def components(self) -> DataFrame:
        from pandas import DataFrame
        components_df = DataFrame({col: getattr(self._parent.array, f'_dt_{col}') for col in ['days', 'hours', 'minutes', 'seconds', 'milliseconds', 'microseconds', 'nanoseconds']})
        return components_df