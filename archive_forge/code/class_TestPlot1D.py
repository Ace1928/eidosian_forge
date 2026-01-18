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
class TestPlot1D(PlotTestCase):

    @pytest.fixture(autouse=True)
    def setUp(self) -> None:
        d = [0, 1.1, 0, 2]
        self.darray = DataArray(d, coords={'period': range(len(d))}, dims='period')
        self.darray.period.attrs['units'] = 's'

    def test_xlabel_is_index_name(self) -> None:
        self.darray.plot()
        assert 'period [s]' == plt.gca().get_xlabel()

    def test_no_label_name_on_x_axis(self) -> None:
        self.darray.plot(y='period')
        assert '' == plt.gca().get_xlabel()

    def test_no_label_name_on_y_axis(self) -> None:
        self.darray.plot()
        assert '' == plt.gca().get_ylabel()

    def test_ylabel_is_data_name(self) -> None:
        self.darray.name = 'temperature'
        self.darray.attrs['units'] = 'degrees_Celsius'
        self.darray.plot()
        assert 'temperature [degrees_Celsius]' == plt.gca().get_ylabel()

    def test_xlabel_is_data_name(self) -> None:
        self.darray.name = 'temperature'
        self.darray.attrs['units'] = 'degrees_Celsius'
        self.darray.plot(y='period')
        assert 'temperature [degrees_Celsius]' == plt.gca().get_xlabel()

    def test_format_string(self) -> None:
        self.darray.plot.line('ro')

    def test_can_pass_in_axis(self) -> None:
        self.pass_in_axis(self.darray.plot.line)

    def test_nonnumeric_index(self) -> None:
        a = DataArray([1, 2, 3], {'letter': ['a', 'b', 'c']}, dims='letter')
        a.plot.line()

    def test_primitive_returned(self) -> None:
        p = self.darray.plot.line()
        assert isinstance(p[0], mpl.lines.Line2D)

    @pytest.mark.slow
    def test_plot_nans(self) -> None:
        self.darray[1] = np.nan
        self.darray.plot.line()

    def test_dates_are_concise(self) -> None:
        import matplotlib.dates as mdates
        time = pd.date_range('2000-01-01', '2000-01-10')
        a = DataArray(np.arange(len(time)), [('t', time)])
        a.plot.line()
        ax = plt.gca()
        assert isinstance(ax.xaxis.get_major_locator(), mdates.AutoDateLocator)
        assert isinstance(ax.xaxis.get_major_formatter(), mdates.ConciseDateFormatter)

    def test_xyincrease_false_changes_axes(self) -> None:
        self.darray.plot.line(xincrease=False, yincrease=False)
        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()
        diffs = (xlim[1] - xlim[0], ylim[1] - ylim[0])
        assert all((x < 0 for x in diffs))

    def test_slice_in_title(self) -> None:
        self.darray.coords['d'] = 10.009
        self.darray.plot.line()
        title = plt.gca().get_title()
        assert 'd = 10.01' == title

    def test_slice_in_title_single_item_array(self) -> None:
        """Edge case for data of shape (1, N) or (N, 1)."""
        darray = self.darray.expand_dims({'d': np.array([10.009])})
        darray.plot.line(x='period')
        title = plt.gca().get_title()
        assert 'd = 10.01' == title