from __future__ import annotations
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_allclose, assert_array_equal, mock
from xarray.tests import assert_identical as assert_identical_
class Other:

    def __array_ufunc__(self, *args, **kwargs):
        return 'other'