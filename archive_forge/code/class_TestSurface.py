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
class TestSurface(Common2dMixin, PlotTestCase):
    plotfunc = staticmethod(xplt.surface)
    subplot_kws = {'projection': '3d'}

    @pytest.mark.xfail(reason='Failing inside matplotlib. Should probably be fixed upstream because other plot functions can handle it. Remove this test when it works, already in Common2dMixin')
    def test_dates_are_concise(self) -> None:
        import matplotlib.dates as mdates
        time = pd.date_range('2000-01-01', '2000-01-10')
        a = DataArray(np.random.randn(2, len(time)), [('xx', [1, 2]), ('t', time)])
        self.plotfunc(a, x='t')
        ax = plt.gca()
        assert isinstance(ax.xaxis.get_major_locator(), mdates.AutoDateLocator)
        assert isinstance(ax.xaxis.get_major_formatter(), mdates.ConciseDateFormatter)

    def test_primitive_artist_returned(self) -> None:
        artist = self.plotmethod()
        assert isinstance(artist, mpl_toolkits.mplot3d.art3d.Poly3DCollection)

    @pytest.mark.slow
    def test_2d_coord_names(self) -> None:
        self.plotmethod(x='x2d', y='y2d')
        ax = plt.gca()
        assert isinstance(ax, mpl_toolkits.mplot3d.axes3d.Axes3D)
        assert 'x2d' == ax.get_xlabel()
        assert 'y2d' == ax.get_ylabel()
        assert f'{self.darray.long_name} [{self.darray.units}]' == ax.get_zlabel()

    def test_xyincrease_false_changes_axes(self) -> None:
        pytest.skip('does not make sense for surface plots')

    def test_xyincrease_true_changes_axes(self) -> None:
        pytest.skip('does not make sense for surface plots')

    def test_can_pass_in_axis(self) -> None:
        self.pass_in_axis(self.plotmethod, subplot_kw={'projection': '3d'})

    def test_default_cmap(self) -> None:
        pytest.skip('does not make sense for surface plots')

    def test_diverging_color_limits(self) -> None:
        pytest.skip('does not make sense for surface plots')

    def test_colorbar_kwargs(self) -> None:
        pytest.skip('does not make sense for surface plots')

    def test_cmap_and_color_both(self) -> None:
        pytest.skip('does not make sense for surface plots')

    def test_seaborn_palette_as_cmap(self) -> None:
        with pytest.raises(ValueError):
            super().test_seaborn_palette_as_cmap()

    @pytest.mark.filterwarnings('ignore:tight_layout cannot')
    def test_convenient_facetgrid(self) -> None:
        a = easy_array((10, 15, 4))
        d = DataArray(a, dims=['y', 'x', 'z'])
        g = self.plotfunc(d, x='x', y='y', col='z', col_wrap=2)
        assert_array_equal(g.axs.shape, [2, 2])
        for (y, x), ax in np.ndenumerate(g.axs):
            assert ax.has_data()
            assert 'y' == ax.get_ylabel()
            assert 'x' == ax.get_xlabel()
        g = self.plotfunc(d, col='z', col_wrap=2)
        assert_array_equal(g.axs.shape, [2, 2])
        for (y, x), ax in np.ndenumerate(g.axs):
            assert ax.has_data()
            assert 'y' == ax.get_ylabel()
            assert 'x' == ax.get_xlabel()

    def test_viridis_cmap(self) -> None:
        return super().test_viridis_cmap()

    def test_can_change_default_cmap(self) -> None:
        return super().test_can_change_default_cmap()

    def test_colorbar_default_label(self) -> None:
        return super().test_colorbar_default_label()

    def test_facetgrid_map_only_appends_mappables(self) -> None:
        return super().test_facetgrid_map_only_appends_mappables()