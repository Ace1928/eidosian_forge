from __future__ import annotations
import contextlib
import inspect
import math
from collections.abc import Hashable
from copy import copy
from datetime import date, datetime, timedelta
from typing import Any, Callable, Literal
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xarray.plot as xplt
from xarray import DataArray, Dataset
from xarray.namedarray.utils import module_available
from xarray.plot.dataarray_plot import _infer_interval_breaks
from xarray.plot.dataset_plot import _infer_meta_data
from xarray.plot.utils import (
from xarray.tests import (
@requires_cftime
@pytest.mark.skipif(has_nc_time_axis, reason='nc_time_axis is installed')
class TestNcAxisNotInstalled(PlotTestCase):

    @pytest.fixture(autouse=True)
    def setUp(self) -> None:
        """
        Create a DataArray with a time-axis that contains cftime.datetime
        objects.
        """
        month = np.arange(1, 13, 1)
        data = np.sin(2 * np.pi * month / 12.0)
        darray = DataArray(data, dims=['time'])
        darray.coords['time'] = xr.cftime_range(start='2017', periods=12, freq='1M', calendar='noleap')
        self.darray = darray

    def test_ncaxis_notinstalled_line_plot(self) -> None:
        with pytest.raises(ImportError, match='optional `nc-time-axis`'):
            self.darray.plot.line()