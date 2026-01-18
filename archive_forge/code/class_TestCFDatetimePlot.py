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
@pytest.mark.filterwarnings('ignore:setting an array element with a sequence')
@requires_cftime
@pytest.mark.skipif(not has_nc_time_axis, reason='nc_time_axis is not installed')
class TestCFDatetimePlot(PlotTestCase):

    @pytest.fixture(autouse=True)
    def setUp(self) -> None:
        """
        Create a DataArray with a time-axis that contains cftime.datetime
        objects.
        """
        data = np.random.rand(4, 12)
        time = xr.cftime_range(start='2017', periods=12, freq='1ME', calendar='noleap')
        darray = DataArray(data, dims=['x', 'time'])
        darray.coords['time'] = time
        self.darray = darray

    def test_cfdatetime_line_plot(self) -> None:
        self.darray.isel(x=0).plot.line()

    def test_cfdatetime_pcolormesh_plot(self) -> None:
        self.darray.plot.pcolormesh()

    def test_cfdatetime_contour_plot(self) -> None:
        self.darray.plot.contour()