from __future__ import annotations
import warnings
from typing import TYPE_CHECKING, Generic
import numpy as np
import pandas as pd
from xarray.coding.times import infer_calendar_name
from xarray.core import duck_array_ops
from xarray.core.common import (
from xarray.core.types import T_DataArray
from xarray.core.variable import IndexVariable
from xarray.namedarray.utils import is_duck_dask_array
class CombinedDatetimelikeAccessor(DatetimeAccessor[T_DataArray], TimedeltaAccessor[T_DataArray]):

    def __new__(cls, obj: T_DataArray) -> CombinedDatetimelikeAccessor:
        if not _contains_datetime_like_objects(obj.variable):
            raise AttributeError("'.dt' accessor only available for DataArray with datetime64 timedelta64 dtype or for arrays containing cftime datetime objects.")
        if is_np_timedelta_like(obj.dtype):
            return TimedeltaAccessor(obj)
        else:
            return DatetimeAccessor(obj)