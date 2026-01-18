from __future__ import annotations
import datetime as dt
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy import array, nan
from xarray import DataArray, Dataset, cftime_range, concat
from xarray.core import dtypes, duck_array_ops
from xarray.core.duck_array_ops import (
from xarray.namedarray.pycompat import array_type
from xarray.testing import assert_allclose, assert_equal, assert_identical
from xarray.tests import (
@requires_dask
class TestDaskOps(TestOps):

    @pytest.fixture(autouse=True)
    def setUp(self):
        import dask.array
        self.x = dask.array.from_array([[[nan, nan, 2.0, nan], [nan, 5.0, 6.0, nan], [8.0, 9.0, 10.0, nan]], [[nan, 13.0, 14.0, 15.0], [nan, 17.0, 18.0, nan], [nan, 21.0, nan, nan]]], chunks=(2, 1, 2))