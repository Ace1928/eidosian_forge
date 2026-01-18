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
class TestPlotHistogram(PlotTestCase):

    @pytest.fixture(autouse=True)
    def setUp(self) -> None:
        self.darray = DataArray(easy_array((2, 3, 4)))

    def test_3d_array(self) -> None:
        self.darray.plot.hist()

    def test_xlabel_uses_name(self) -> None:
        self.darray.name = 'testpoints'
        self.darray.attrs['units'] = 'testunits'
        self.darray.plot.hist()
        assert 'testpoints [testunits]' == plt.gca().get_xlabel()

    def test_title_is_histogram(self) -> None:
        self.darray.coords['d'] = 10
        self.darray.plot.hist()
        assert 'd = 10' == plt.gca().get_title()

    def test_can_pass_in_kwargs(self) -> None:
        nbins = 5
        self.darray.plot.hist(bins=nbins)
        assert nbins == len(plt.gca().patches)

    def test_can_pass_in_axis(self) -> None:
        self.pass_in_axis(self.darray.plot.hist)

    def test_primitive_returned(self) -> None:
        n, bins, patches = self.darray.plot.hist()
        assert isinstance(n, np.ndarray)
        assert isinstance(bins, np.ndarray)
        assert isinstance(patches, mpl.container.BarContainer)
        assert isinstance(patches[0], mpl.patches.Rectangle)

    @pytest.mark.slow
    def test_plot_nans(self) -> None:
        self.darray[0, 0, 0] = np.nan
        self.darray.plot.hist()

    def test_hist_coord_with_interval(self) -> None:
        self.darray.groupby_bins('dim_0', [-1, 0, 1, 2]).mean(...).plot.hist(range=(-1, 2))