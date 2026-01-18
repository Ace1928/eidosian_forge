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
class TestFacetGrid(PlotTestCase):

    @pytest.fixture(autouse=True)
    def setUp(self) -> None:
        d = easy_array((10, 15, 3))
        self.darray = DataArray(d, dims=['y', 'x', 'z'], coords={'z': ['a', 'b', 'c']})
        self.g = xplt.FacetGrid(self.darray, col='z')

    @pytest.mark.slow
    def test_no_args(self) -> None:
        self.g.map_dataarray(xplt.contourf, 'x', 'y')
        alltxt = text_in_fig()
        assert 'None' not in alltxt
        for ax in self.g.axs.flat:
            assert ax.has_data()

    @pytest.mark.slow
    def test_names_appear_somewhere(self) -> None:
        self.darray.name = 'testvar'
        self.g.map_dataarray(xplt.contourf, 'x', 'y')
        for k, ax in zip('abc', self.g.axs.flat):
            assert f'z = {k}' == ax.get_title()
        alltxt = text_in_fig()
        assert self.darray.name in alltxt
        for label in ['x', 'y']:
            assert label in alltxt

    @pytest.mark.slow
    def test_text_not_super_long(self) -> None:
        self.darray.coords['z'] = [100 * letter for letter in 'abc']
        g = xplt.FacetGrid(self.darray, col='z')
        g.map_dataarray(xplt.contour, 'x', 'y')
        alltxt = text_in_fig()
        maxlen = max((len(txt) for txt in alltxt))
        assert maxlen < 50
        t0 = g.axs[0, 0].get_title()
        assert t0.endswith('...')

    @pytest.mark.slow
    def test_colorbar(self) -> None:
        vmin = self.darray.values.min()
        vmax = self.darray.values.max()
        expected = np.array((vmin, vmax))
        self.g.map_dataarray(xplt.imshow, 'x', 'y')
        for image in plt.gcf().findobj(mpl.image.AxesImage):
            assert isinstance(image, mpl.image.AxesImage)
            clim = np.array(image.get_clim())
            assert np.allclose(expected, clim)
        assert 1 == len(find_possible_colorbars())

    def test_colorbar_scatter(self) -> None:
        ds = Dataset({'a': (('x', 'y'), np.arange(4).reshape(2, 2))})
        fg: xplt.FacetGrid = ds.plot.scatter(x='a', y='a', row='x', hue='a')
        cbar = fg.cbar
        assert cbar is not None
        assert hasattr(cbar, 'vmin')
        assert cbar.vmin == 0
        assert hasattr(cbar, 'vmax')
        assert cbar.vmax == 3

    @pytest.mark.slow
    def test_empty_cell(self) -> None:
        g = xplt.FacetGrid(self.darray, col='z', col_wrap=2)
        g.map_dataarray(xplt.imshow, 'x', 'y')
        bottomright = g.axs[-1, -1]
        assert not bottomright.has_data()
        assert not bottomright.get_visible()

    @pytest.mark.slow
    def test_norow_nocol_error(self) -> None:
        with pytest.raises(ValueError, match='[Rr]ow'):
            xplt.FacetGrid(self.darray)

    @pytest.mark.slow
    def test_groups(self) -> None:
        self.g.map_dataarray(xplt.imshow, 'x', 'y')
        upperleft_dict = self.g.name_dicts[0, 0]
        upperleft_array = self.darray.loc[upperleft_dict]
        z0 = self.darray.isel(z=0)
        assert_equal(upperleft_array, z0)

    @pytest.mark.slow
    def test_float_index(self) -> None:
        self.darray.coords['z'] = [0.1, 0.2, 0.4]
        g = xplt.FacetGrid(self.darray, col='z')
        g.map_dataarray(xplt.imshow, 'x', 'y')

    @pytest.mark.slow
    def test_nonunique_index_error(self) -> None:
        self.darray.coords['z'] = [0.1, 0.2, 0.2]
        with pytest.raises(ValueError, match='[Uu]nique'):
            xplt.FacetGrid(self.darray, col='z')

    @pytest.mark.slow
    def test_robust(self) -> None:
        z = np.zeros((20, 20, 2))
        darray = DataArray(z, dims=['y', 'x', 'z'])
        darray[:, :, 1] = 1
        darray[2, 0, 0] = -1000
        darray[3, 0, 0] = 1000
        g = xplt.FacetGrid(darray, col='z')
        g.map_dataarray(xplt.imshow, 'x', 'y', robust=True)
        numbers = set()
        alltxt = text_in_fig()
        for txt in alltxt:
            try:
                numbers.add(float(txt))
            except ValueError:
                pass
        largest = max((abs(x) for x in numbers))
        assert largest < 21

    @pytest.mark.slow
    def test_can_set_vmin_vmax(self) -> None:
        vmin, vmax = (50.0, 1000.0)
        expected = np.array((vmin, vmax))
        self.g.map_dataarray(xplt.imshow, 'x', 'y', vmin=vmin, vmax=vmax)
        for image in plt.gcf().findobj(mpl.image.AxesImage):
            assert isinstance(image, mpl.image.AxesImage)
            clim = np.array(image.get_clim())
            assert np.allclose(expected, clim)

    @pytest.mark.slow
    def test_vmin_vmax_equal(self) -> None:
        fg = self.g.map_dataarray(xplt.imshow, 'x', 'y', vmin=50, vmax=50)
        for mappable in fg._mappables:
            assert mappable.norm.vmin != mappable.norm.vmax

    @pytest.mark.slow
    @pytest.mark.filterwarnings('ignore')
    def test_can_set_norm(self) -> None:
        norm = mpl.colors.SymLogNorm(0.1)
        self.g.map_dataarray(xplt.imshow, 'x', 'y', norm=norm)
        for image in plt.gcf().findobj(mpl.image.AxesImage):
            assert isinstance(image, mpl.image.AxesImage)
            assert image.norm is norm

    @pytest.mark.slow
    def test_figure_size(self) -> None:
        assert_array_equal(self.g.fig.get_size_inches(), (10, 3))
        g = xplt.FacetGrid(self.darray, col='z', size=6)
        assert_array_equal(g.fig.get_size_inches(), (19, 6))
        g = self.darray.plot.imshow(col='z', size=6)
        assert_array_equal(g.fig.get_size_inches(), (19, 6))
        g = xplt.FacetGrid(self.darray, col='z', size=4, aspect=0.5)
        assert_array_equal(g.fig.get_size_inches(), (7, 4))
        g = xplt.FacetGrid(self.darray, col='z', figsize=(9, 4))
        assert_array_equal(g.fig.get_size_inches(), (9, 4))
        with pytest.raises(ValueError, match='cannot provide both'):
            g = xplt.plot(self.darray, row=2, col='z', figsize=(6, 4), size=6)
        with pytest.raises(ValueError, match="Can't use"):
            g = xplt.plot(self.darray, row=2, col='z', ax=plt.gca(), size=6)

    @pytest.mark.slow
    def test_num_ticks(self) -> None:
        nticks = 99
        maxticks = nticks + 1
        self.g.map_dataarray(xplt.imshow, 'x', 'y')
        self.g.set_ticks(max_xticks=nticks, max_yticks=nticks)
        for ax in self.g.axs.flat:
            xticks = len(ax.get_xticks())
            yticks = len(ax.get_yticks())
            assert xticks <= maxticks
            assert yticks <= maxticks
            assert xticks >= nticks / 2.0
            assert yticks >= nticks / 2.0

    @pytest.mark.slow
    def test_map(self) -> None:
        assert self.g._finalized is False
        self.g.map(plt.contourf, 'x', 'y', ...)
        assert self.g._finalized is True
        self.g.map(lambda: None)

    @pytest.mark.slow
    def test_map_dataset(self) -> None:
        g = xplt.FacetGrid(self.darray.to_dataset(name='foo'), col='z')
        g.map(plt.contourf, 'x', 'y', 'foo')
        alltxt = text_in_fig()
        for label in ['x', 'y']:
            assert label in alltxt
        assert 'None' not in alltxt
        assert 'foo' not in alltxt
        assert 0 == len(find_possible_colorbars())
        g.add_colorbar(label='colors!')
        assert 'colors!' in text_in_fig()
        assert 1 == len(find_possible_colorbars())

    @pytest.mark.slow
    def test_set_axis_labels(self) -> None:
        g = self.g.map_dataarray(xplt.contourf, 'x', 'y')
        g.set_axis_labels('longitude', 'latitude')
        alltxt = text_in_fig()
        for label in ['longitude', 'latitude']:
            assert label in alltxt

    @pytest.mark.slow
    def test_facetgrid_colorbar(self) -> None:
        a = easy_array((10, 15, 4))
        d = DataArray(a, dims=['y', 'x', 'z'], name='foo')
        d.plot.imshow(x='x', y='y', col='z')
        assert 1 == len(find_possible_colorbars())
        d.plot.imshow(x='x', y='y', col='z', add_colorbar=True)
        assert 1 == len(find_possible_colorbars())
        d.plot.imshow(x='x', y='y', col='z', add_colorbar=False)
        assert 0 == len(find_possible_colorbars())

    @pytest.mark.slow
    def test_facetgrid_polar(self) -> None:
        self.darray.plot.pcolormesh(col='z', subplot_kws=dict(projection='polar'), sharex=False, sharey=False)