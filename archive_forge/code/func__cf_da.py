from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core.missing import (
from xarray.namedarray.pycompat import array_type
from xarray.tests import (
def _cf_da(calendar, freq='1D'):
    times = xr.cftime_range(start='1970-01-01', freq=freq, periods=10, calendar=calendar)
    values = np.arange(10)
    return xr.DataArray(values, dims=('time',), coords={'time': times})