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
def _access_through_series(values, name):
    """Coerce an array of datetime-like values to a pandas Series and
    access requested datetime component
    """
    values_as_series = pd.Series(duck_array_ops.ravel(values), copy=False)
    if name == 'season':
        months = values_as_series.dt.month.values
        field_values = _season_from_months(months)
    elif name == 'total_seconds':
        field_values = values_as_series.dt.total_seconds().values
    elif name == 'isocalendar':
        field_values = values_as_series.dt.isocalendar()
        hasna = any(field_values.year.isnull())
        if hasna:
            field_values = np.dstack([getattr(field_values, name).astype(np.float64, copy=False).values for name in ['year', 'week', 'day']])
        else:
            field_values = np.array(field_values, dtype=np.int64)
        return field_values.T.reshape(3, *values.shape)
    else:
        field_values = getattr(values_as_series.dt, name).values
    return field_values.reshape(values.shape)