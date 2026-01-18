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
@pytest.mark.filterwarnings('ignore:tight_layout cannot')
class TestFacetGrid4d(PlotTestCase):

    @pytest.fixture(autouse=True)
    def setUp(self) -> None:
        a = easy_array((10, 15, 3, 2))
        darray = DataArray(a, dims=['y', 'x', 'col', 'row'])
        darray.coords['col'] = np.array(['col' + str(x) for x in darray.coords['col'].values])
        darray.coords['row'] = np.array(['row' + str(x) for x in darray.coords['row'].values])
        self.darray = darray

    def test_title_kwargs(self) -> None:
        g = xplt.FacetGrid(self.darray, col='col', row='row')
        g.set_titles(template='{value}', weight='bold')
        for label, ax in zip(self.darray.coords['row'].values, g.axs[:, -1]):
            assert property_in_axes_text('weight', 'bold', label, ax)
        for label, ax in zip(self.darray.coords['col'].values, g.axs[0, :]):
            assert property_in_axes_text('weight', 'bold', label, ax)

    @pytest.mark.slow
    def test_default_labels(self) -> None:
        g = xplt.FacetGrid(self.darray, col='col', row='row')
        assert (2, 3) == g.axs.shape
        g.map_dataarray(xplt.imshow, 'x', 'y')
        for label, ax in zip(self.darray.coords['row'].values, g.axs[:, -1]):
            assert substring_in_axes(label, ax)
        for label, ax in zip(self.darray.coords['col'].values, g.axs[0, :]):
            assert substring_in_axes(label, ax)
        g.set_titles('abc={value}')
        for label, ax in zip(self.darray.coords['row'].values, g.axs[:, -1]):
            assert substring_in_axes(f'abc={label}', ax)
            assert substring_not_in_axes('row=', ax)
        for label, ax in zip(self.darray.coords['col'].values, g.axs[0, :]):
            assert substring_in_axes(f'abc={label}', ax)
            assert substring_not_in_axes('col=', ax)