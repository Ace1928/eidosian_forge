from __future__ import annotations
import math
import re
import warnings
from datetime import timedelta
import numpy as np
import pandas as pd
from packaging.version import Version
from xarray.coding.times import (
from xarray.core.common import _contains_cftime_datetimes
from xarray.core.options import OPTIONS
from xarray.core.utils import is_scalar
def _partial_date_slice(self, resolution, parsed):
    """Adapted from
        pandas.tseries.index.DatetimeIndex._partial_date_slice

        Note that when using a CFTimeIndex, if a partial-date selection
        returns a single element, it will never be converted to a scalar
        coordinate; this is in slight contrast to the behavior when using
        a DatetimeIndex, which sometimes will return a DataArray with a scalar
        coordinate depending on the resolution of the datetimes used in
        defining the index.  For example:

        >>> from cftime import DatetimeNoLeap
        >>> da = xr.DataArray(
        ...     [1, 2],
        ...     coords=[[DatetimeNoLeap(2001, 1, 1), DatetimeNoLeap(2001, 2, 1)]],
        ...     dims=["time"],
        ... )
        >>> da.sel(time="2001-01-01")
        <xarray.DataArray (time: 1)> Size: 8B
        array([1])
        Coordinates:
          * time     (time) object 8B 2001-01-01 00:00:00
        >>> da = xr.DataArray(
        ...     [1, 2],
        ...     coords=[[pd.Timestamp(2001, 1, 1), pd.Timestamp(2001, 2, 1)]],
        ...     dims=["time"],
        ... )
        >>> da.sel(time="2001-01-01")
        <xarray.DataArray ()> Size: 8B
        array(1)
        Coordinates:
            time     datetime64[ns] 8B 2001-01-01
        >>> da = xr.DataArray(
        ...     [1, 2],
        ...     coords=[[pd.Timestamp(2001, 1, 1, 1), pd.Timestamp(2001, 2, 1)]],
        ...     dims=["time"],
        ... )
        >>> da.sel(time="2001-01-01")
        <xarray.DataArray (time: 1)> Size: 8B
        array([1])
        Coordinates:
          * time     (time) datetime64[ns] 8B 2001-01-01T01:00:00
        """
    start, end = _parsed_string_to_bounds(self.date_type, resolution, parsed)
    times = self._data
    if self.is_monotonic_increasing:
        if len(times) and (start < times[0] and end < times[0] or (start > times[-1] and end > times[-1])):
            raise KeyError
        left = times.searchsorted(start, side='left')
        right = times.searchsorted(end, side='right')
        return slice(left, right)
    lhs_mask = times >= start
    rhs_mask = times <= end
    return np.flatnonzero(lhs_mask & rhs_mask)