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
def _season_from_months(months):
    """Compute season (DJF, MAM, JJA, SON) from month ordinal"""
    seasons = np.array(['DJF', 'MAM', 'JJA', 'SON', 'nan'])
    months = np.asarray(months)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='invalid value encountered in floor_divide')
        warnings.filterwarnings('ignore', message='invalid value encountered in remainder')
        idx = months // 3 % 4
    idx[np.isnan(idx)] = 4
    return seasons[idx.astype(int)]