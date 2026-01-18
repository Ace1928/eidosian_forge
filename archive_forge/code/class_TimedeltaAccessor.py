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
class TimedeltaAccessor(TimeAccessor[T_DataArray]):
    """Access Timedelta fields for DataArrays with Timedelta-like dtypes.

    Fields can be accessed through the `.dt` attribute for applicable DataArrays.

    Examples
    --------
    >>> dates = pd.timedelta_range(start="1 day", freq="6h", periods=20)
    >>> ts = xr.DataArray(dates, dims=("time"))
    >>> ts
    <xarray.DataArray (time: 20)> Size: 160B
    array([ 86400000000000, 108000000000000, 129600000000000, 151200000000000,
           172800000000000, 194400000000000, 216000000000000, 237600000000000,
           259200000000000, 280800000000000, 302400000000000, 324000000000000,
           345600000000000, 367200000000000, 388800000000000, 410400000000000,
           432000000000000, 453600000000000, 475200000000000, 496800000000000],
          dtype='timedelta64[ns]')
    Coordinates:
      * time     (time) timedelta64[ns] 160B 1 days 00:00:00 ... 5 days 18:00:00
    >>> ts.dt  # doctest: +ELLIPSIS
    <xarray.core.accessor_dt.TimedeltaAccessor object at 0x...>
    >>> ts.dt.days
    <xarray.DataArray 'days' (time: 20)> Size: 160B
    array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5])
    Coordinates:
      * time     (time) timedelta64[ns] 160B 1 days 00:00:00 ... 5 days 18:00:00
    >>> ts.dt.microseconds
    <xarray.DataArray 'microseconds' (time: 20)> Size: 160B
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Coordinates:
      * time     (time) timedelta64[ns] 160B 1 days 00:00:00 ... 5 days 18:00:00
    >>> ts.dt.seconds
    <xarray.DataArray 'seconds' (time: 20)> Size: 160B
    array([    0, 21600, 43200, 64800,     0, 21600, 43200, 64800,     0,
           21600, 43200, 64800,     0, 21600, 43200, 64800,     0, 21600,
           43200, 64800])
    Coordinates:
      * time     (time) timedelta64[ns] 160B 1 days 00:00:00 ... 5 days 18:00:00
    >>> ts.dt.total_seconds()
    <xarray.DataArray 'total_seconds' (time: 20)> Size: 160B
    array([ 86400., 108000., 129600., 151200., 172800., 194400., 216000.,
           237600., 259200., 280800., 302400., 324000., 345600., 367200.,
           388800., 410400., 432000., 453600., 475200., 496800.])
    Coordinates:
      * time     (time) timedelta64[ns] 160B 1 days 00:00:00 ... 5 days 18:00:00
    """

    @property
    def days(self) -> T_DataArray:
        """Number of days for each element"""
        return self._date_field('days', np.int64)

    @property
    def seconds(self) -> T_DataArray:
        """Number of seconds (>= 0 and less than 1 day) for each element"""
        return self._date_field('seconds', np.int64)

    @property
    def microseconds(self) -> T_DataArray:
        """Number of microseconds (>= 0 and less than 1 second) for each element"""
        return self._date_field('microseconds', np.int64)

    @property
    def nanoseconds(self) -> T_DataArray:
        """Number of nanoseconds (>= 0 and less than 1 microsecond) for each element"""
        return self._date_field('nanoseconds', np.int64)

    def total_seconds(self) -> T_DataArray:
        """Total duration of each element expressed in seconds."""
        return self._date_field('total_seconds', np.float64)