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
@requires_matplotlib
class TestDatasetStreamplotPlots(PlotTestCase):

    @pytest.fixture(autouse=True)
    def setUp(self) -> None:
        das = [DataArray(np.random.randn(3, 3, 2, 2), dims=['x', 'y', 'row', 'col'], coords=[range(k) for k in [3, 3, 2, 2]]) for _ in [1, 2]]
        ds = Dataset({'u': das[0], 'v': das[1]})
        ds.x.attrs['units'] = 'xunits'
        ds.y.attrs['units'] = 'yunits'
        ds.col.attrs['units'] = 'colunits'
        ds.row.attrs['units'] = 'rowunits'
        ds.u.attrs['units'] = 'uunits'
        ds.v.attrs['units'] = 'vunits'
        ds['mag'] = np.hypot(ds.u, ds.v)
        self.ds = ds

    def test_streamline(self) -> None:
        with figure_context():
            hdl = self.ds.isel(row=0, col=0).plot.streamplot(x='x', y='y', u='u', v='v')
            assert isinstance(hdl, mpl.collections.LineCollection)
        with pytest.raises(ValueError, match='specify x, y, u, v'):
            self.ds.isel(row=0, col=0).plot.streamplot(x='x', y='y', u='u')
        with pytest.raises(ValueError, match='hue_style'):
            self.ds.isel(row=0, col=0).plot.streamplot(x='x', y='y', u='u', v='v', hue='mag', hue_style='discrete')

    def test_facetgrid(self) -> None:
        with figure_context():
            fg = self.ds.plot.streamplot(x='x', y='y', u='u', v='v', row='row', col='col', hue='mag')
            for handle in fg._mappables:
                assert isinstance(handle, mpl.collections.LineCollection)
        with figure_context():
            fg = self.ds.plot.streamplot(x='x', y='y', u='u', v='v', row='row', col='col', hue='mag', add_guide=False)