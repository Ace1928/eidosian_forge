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
class TestContour(Common2dMixin, PlotTestCase):
    plotfunc = staticmethod(xplt.contour)

    @staticmethod
    def _color_as_tuple(c: Any) -> tuple[Any, Any, Any]:
        return (c[0], c[1], c[2])

    def test_colors(self) -> None:
        artist = self.plotmethod(colors='k')
        assert artist.cmap.colors[0] == 'k'
        artist = self.plotmethod(colors=['k', 'b'])
        assert self._color_as_tuple(artist.cmap.colors[1]) == (0.0, 0.0, 1.0)
        artist = self.darray.plot.contour(levels=[-0.5, 0.0, 0.5, 1.0], colors=['k', 'r', 'w', 'b'])
        assert self._color_as_tuple(artist.cmap.colors[1]) == (1.0, 0.0, 0.0)
        assert self._color_as_tuple(artist.cmap.colors[2]) == (1.0, 1.0, 1.0)
        assert self._color_as_tuple(artist.cmap._rgba_over) == (0.0, 0.0, 1.0)

    def test_colors_np_levels(self) -> None:
        levels = np.array([-0.5, 0.0, 0.5, 1.0])
        artist = self.darray.plot.contour(levels=levels, colors=['k', 'r', 'w', 'b'])
        cmap = artist.cmap
        assert isinstance(cmap, mpl.colors.ListedColormap)
        colors = cmap.colors
        assert isinstance(colors, list)
        assert self._color_as_tuple(colors[1]) == (1.0, 0.0, 0.0)
        assert self._color_as_tuple(colors[2]) == (1.0, 1.0, 1.0)
        assert hasattr(cmap, '_rgba_over')
        assert self._color_as_tuple(cmap._rgba_over) == (0.0, 0.0, 1.0)

    def test_cmap_and_color_both(self) -> None:
        with pytest.raises(ValueError):
            self.plotmethod(colors='k', cmap='RdBu')

    def list_of_colors_in_cmap_raises_error(self) -> None:
        with pytest.raises(ValueError, match='list of colors'):
            self.plotmethod(cmap=['k', 'b'])

    @pytest.mark.slow
    def test_2d_coord_names(self) -> None:
        self.plotmethod(x='x2d', y='y2d')
        ax = plt.gca()
        assert 'x2d' == ax.get_xlabel()
        assert 'y2d' == ax.get_ylabel()

    def test_single_level(self) -> None:
        self.plotmethod(levels=[0.1])
        self.plotmethod(levels=1)