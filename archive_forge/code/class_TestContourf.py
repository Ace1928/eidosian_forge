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
@pytest.mark.slow
class TestContourf(Common2dMixin, PlotTestCase):
    plotfunc = staticmethod(xplt.contourf)

    @pytest.mark.slow
    def test_contourf_called(self) -> None:
        assert not self.contourf_called(self.darray.plot.imshow)
        assert self.contourf_called(self.darray.plot.contourf)

    def test_primitive_artist_returned(self) -> None:
        artist = self.plotmethod()
        assert isinstance(artist, mpl.contour.QuadContourSet)

    @pytest.mark.slow
    def test_extend(self) -> None:
        artist = self.plotmethod()
        assert artist.extend == 'neither'
        self.darray[0, 0] = -100
        self.darray[-1, -1] = 100
        artist = self.plotmethod(robust=True)
        assert artist.extend == 'both'
        self.darray[0, 0] = 0
        self.darray[-1, -1] = 0
        artist = self.plotmethod(vmin=-0, vmax=10)
        assert artist.extend == 'min'
        artist = self.plotmethod(vmin=-10, vmax=0)
        assert artist.extend == 'max'

    @pytest.mark.slow
    def test_2d_coord_names(self) -> None:
        self.plotmethod(x='x2d', y='y2d')
        ax = plt.gca()
        assert 'x2d' == ax.get_xlabel()
        assert 'y2d' == ax.get_ylabel()

    @pytest.mark.slow
    def test_levels(self) -> None:
        artist = self.plotmethod(levels=[-0.5, -0.4, 0.1])
        assert artist.extend == 'both'
        artist = self.plotmethod(levels=3)
        assert artist.extend == 'neither'