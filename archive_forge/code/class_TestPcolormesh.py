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
class TestPcolormesh(Common2dMixin, PlotTestCase):
    plotfunc = staticmethod(xplt.pcolormesh)

    def test_primitive_artist_returned(self) -> None:
        artist = self.plotmethod()
        assert isinstance(artist, mpl.collections.QuadMesh)

    def test_everything_plotted(self) -> None:
        artist = self.plotmethod()
        assert artist.get_array().size == self.darray.size

    @pytest.mark.slow
    def test_2d_coord_names(self) -> None:
        self.plotmethod(x='x2d', y='y2d')
        ax = plt.gca()
        assert 'x2d' == ax.get_xlabel()
        assert 'y2d' == ax.get_ylabel()

    def test_dont_infer_interval_breaks_for_cartopy(self) -> None:
        ax = plt.gca()
        setattr(ax, 'projection', True)
        artist = self.plotmethod(x='x2d', y='y2d', ax=ax)
        assert isinstance(artist, mpl.collections.QuadMesh)
        arr = artist.get_array()
        assert arr is not None
        assert arr.size <= self.darray.size