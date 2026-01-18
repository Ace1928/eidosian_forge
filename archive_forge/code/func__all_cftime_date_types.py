from __future__ import annotations
import warnings
from datetime import timedelta
from itertools import product
import numpy as np
import pandas as pd
import pytest
from pandas.errors import OutOfBoundsDatetime
from xarray import (
from xarray.coding.times import (
from xarray.coding.variables import SerializationWarning
from xarray.conventions import _update_bounds_attributes, cf_encoder
from xarray.core.common import contains_cftime_datetimes
from xarray.core.utils import is_duck_dask_array
from xarray.testing import assert_equal, assert_identical
from xarray.tests import (
def _all_cftime_date_types():
    import cftime
    return {'noleap': cftime.DatetimeNoLeap, '365_day': cftime.DatetimeNoLeap, '360_day': cftime.Datetime360Day, 'julian': cftime.DatetimeJulian, 'all_leap': cftime.DatetimeAllLeap, '366_day': cftime.DatetimeAllLeap, 'gregorian': cftime.DatetimeGregorian, 'proleptic_gregorian': cftime.DatetimeProlepticGregorian}