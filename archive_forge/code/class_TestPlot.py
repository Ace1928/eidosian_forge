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
class TestPlot(PlotTestCase):

    @pytest.fixture(autouse=True)
    def setup_array(self) -> None:
        self.darray = DataArray(easy_array((2, 3, 4)))

    def test_accessor(self) -> None:
        from xarray.plot.accessor import DataArrayPlotAccessor
        assert DataArray.plot is DataArrayPlotAccessor
        assert isinstance(self.darray.plot, DataArrayPlotAccessor)

    def test_label_from_attrs(self) -> None:
        da = self.darray.copy()
        assert '' == label_from_attrs(da)
        da.name = 0
        assert '0' == label_from_attrs(da)
        da.name = 'a'
        da.attrs['units'] = 'a_units'
        da.attrs['long_name'] = 'a_long_name'
        da.attrs['standard_name'] = 'a_standard_name'
        assert 'a_long_name [a_units]' == label_from_attrs(da)
        da.attrs.pop('long_name')
        assert 'a_standard_name [a_units]' == label_from_attrs(da)
        da.attrs.pop('units')
        assert 'a_standard_name' == label_from_attrs(da)
        da.attrs['units'] = 'a_units'
        da.attrs.pop('standard_name')
        assert 'a [a_units]' == label_from_attrs(da)
        da.attrs.pop('units')
        assert 'a' == label_from_attrs(da)
        long_latex_name = '$Ra_s = \\mathrm{mean}(\\epsilon_k) / \\mu M^2_\\infty$'
        da.attrs = dict(long_name=long_latex_name)
        assert label_from_attrs(da) == long_latex_name

    def test1d(self) -> None:
        self.darray[:, 0, 0].plot()
        with pytest.raises(ValueError, match="x must be one of None, 'dim_0'"):
            self.darray[:, 0, 0].plot(x='dim_1')
        with pytest.raises(TypeError, match='complex128'):
            (self.darray[:, 0, 0] + 1j).plot()

    def test_1d_bool(self) -> None:
        xr.ones_like(self.darray[:, 0, 0], dtype=bool).plot()

    def test_1d_x_y_kw(self) -> None:
        z = np.arange(10)
        da = DataArray(np.cos(z), dims=['z'], coords=[z], name='f')
        xy: list[list[None | str]] = [[None, None], [None, 'z'], ['z', None]]
        f, ax = plt.subplots(3, 1)
        for aa, (x, y) in enumerate(xy):
            da.plot(x=x, y=y, ax=ax.flat[aa])
        with pytest.raises(ValueError, match='Cannot specify both'):
            da.plot(x='z', y='z')
        error_msg = "must be one of None, 'z'"
        with pytest.raises(ValueError, match=f'x {error_msg}'):
            da.plot(x='f')
        with pytest.raises(ValueError, match=f'y {error_msg}'):
            da.plot(y='f')

    def test_multiindex_level_as_coord(self) -> None:
        da = xr.DataArray(np.arange(5), dims='x', coords=dict(a=('x', np.arange(5)), b=('x', np.arange(5, 10))))
        da = da.set_index(x=['a', 'b'])
        for x in ['a', 'b']:
            h = da.plot(x=x)[0]
            assert_array_equal(h.get_xdata(), da[x].values)
        for y in ['a', 'b']:
            h = da.plot(y=y)[0]
            assert_array_equal(h.get_ydata(), da[y].values)

    def test_infer_line_data(self) -> None:
        current = DataArray(name='I', data=np.array([5, 8]), dims=['t'], coords={'t': (['t'], np.array([0.1, 0.2])), 'V': (['t'], np.array([100, 200]))})
        line = current.plot.line(x='V')[0]
        assert_array_equal(line.get_xdata(), current.coords['V'].values)
        line = current.plot.line()[0]
        assert_array_equal(line.get_xdata(), current.coords['t'].values)

    def test_line_plot_along_1d_coord(self) -> None:
        x_coord = xr.DataArray(data=[0.1, 0.2], dims=['x'])
        t_coord = xr.DataArray(data=[10, 20], dims=['t'])
        da = xr.DataArray(data=np.array([[0, 1], [5, 9]]), dims=['x', 't'], coords={'x': x_coord, 'time': t_coord})
        line = da.plot(x='time', hue='x')[0]
        assert_array_equal(line.get_xdata(), da.coords['time'].values)
        line = da.plot(y='time', hue='x')[0]
        assert_array_equal(line.get_ydata(), da.coords['time'].values)

    def test_line_plot_wrong_hue(self) -> None:
        da = xr.DataArray(data=np.array([[0, 1], [5, 9]]), dims=['x', 't'])
        with pytest.raises(ValueError, match='hue must be one of'):
            da.plot(x='t', hue='wrong_coord')

    def test_2d_line(self) -> None:
        with pytest.raises(ValueError, match='hue'):
            self.darray[:, :, 0].plot.line()
        self.darray[:, :, 0].plot.line(hue='dim_1')
        self.darray[:, :, 0].plot.line(x='dim_1')
        self.darray[:, :, 0].plot.line(y='dim_1')
        self.darray[:, :, 0].plot.line(x='dim_0', hue='dim_1')
        self.darray[:, :, 0].plot.line(y='dim_0', hue='dim_1')
        with pytest.raises(ValueError, match='Cannot'):
            self.darray[:, :, 0].plot.line(x='dim_1', y='dim_0', hue='dim_1')

    def test_2d_line_accepts_legend_kw(self) -> None:
        self.darray[:, :, 0].plot.line(x='dim_0', add_legend=False)
        assert not plt.gca().get_legend()
        plt.cla()
        self.darray[:, :, 0].plot.line(x='dim_0', add_legend=True)
        assert plt.gca().get_legend()
        assert plt.gca().get_legend().get_title().get_text() == 'dim_1'

    def test_2d_line_accepts_x_kw(self) -> None:
        self.darray[:, :, 0].plot.line(x='dim_0')
        assert plt.gca().get_xlabel() == 'dim_0'
        plt.cla()
        self.darray[:, :, 0].plot.line(x='dim_1')
        assert plt.gca().get_xlabel() == 'dim_1'

    def test_2d_line_accepts_hue_kw(self) -> None:
        self.darray[:, :, 0].plot.line(hue='dim_0')
        assert plt.gca().get_legend().get_title().get_text() == 'dim_0'
        plt.cla()
        self.darray[:, :, 0].plot.line(hue='dim_1')
        assert plt.gca().get_legend().get_title().get_text() == 'dim_1'

    def test_2d_coords_line_plot(self) -> None:
        lon, lat = np.meshgrid(np.linspace(-20, 20, 5), np.linspace(0, 30, 4))
        lon += lat / 10
        lat += lon / 10
        da = xr.DataArray(np.arange(20).reshape(4, 5), dims=['y', 'x'], coords={'lat': (('y', 'x'), lat), 'lon': (('y', 'x'), lon)})
        with figure_context():
            hdl = da.plot.line(x='lon', hue='x')
            assert len(hdl) == 5
        with figure_context():
            hdl = da.plot.line(x='lon', hue='y')
            assert len(hdl) == 4
        with pytest.raises(ValueError, match='For 2D inputs, hue must be a dimension'):
            da.plot.line(x='lon', hue='lat')

    def test_2d_coord_line_plot_coords_transpose_invariant(self) -> None:
        x = np.arange(10)
        y = np.arange(20)
        ds = xr.Dataset(coords={'x': x, 'y': y})
        for z in [ds.y + ds.x, ds.x + ds.y]:
            ds = ds.assign_coords(z=z)
            ds['v'] = ds.x + ds.y
            ds['v'].plot.line(y='z', hue='x')

    def test_2d_before_squeeze(self) -> None:
        a = DataArray(easy_array((1, 5)))
        a.plot()

    def test2d_uniform_calls_imshow(self) -> None:
        assert self.imshow_called(self.darray[:, :, 0].plot.imshow)

    @pytest.mark.slow
    def test2d_nonuniform_calls_contourf(self) -> None:
        a = self.darray[:, :, 0]
        a.coords['dim_1'] = [2, 1, 89]
        assert self.contourf_called(a.plot.contourf)

    def test2d_1d_2d_coordinates_contourf(self) -> None:
        sz = (20, 10)
        depth = easy_array(sz)
        a = DataArray(easy_array(sz), dims=['z', 'time'], coords={'depth': (['z', 'time'], depth), 'time': np.linspace(0, 1, sz[1])})
        a.plot.contourf(x='time', y='depth')
        a.plot.contourf(x='depth', y='time')

    def test2d_1d_2d_coordinates_pcolormesh(self) -> None:
        sz = 10
        y2d, x2d = np.meshgrid(np.arange(sz), np.arange(sz))
        a = DataArray(easy_array((sz, sz)), dims=['x', 'y'], coords={'x2d': (['x', 'y'], x2d), 'y2d': (['x', 'y'], y2d)})
        for x, y in [('x', 'y'), ('y', 'x'), ('x2d', 'y'), ('y', 'x2d'), ('x', 'y2d'), ('y2d', 'x'), ('x2d', 'y2d'), ('y2d', 'x2d')]:
            p = a.plot.pcolormesh(x=x, y=y)
            v = p.get_paths()[0].vertices
            assert isinstance(v, np.ndarray)
            _, unique_counts = np.unique(v[:-1], axis=0, return_counts=True)
            assert np.all(unique_counts == 1)

    def test_str_coordinates_pcolormesh(self) -> None:
        x = DataArray([[1, 2, 3], [4, 5, 6]], dims=('a', 'b'), coords={'a': [1, 2], 'b': ['a', 'b', 'c']})
        x.plot.pcolormesh()
        x.T.plot.pcolormesh()

    def test_contourf_cmap_set(self) -> None:
        a = DataArray(easy_array((4, 4)), dims=['z', 'time'])
        cmap_expected = mpl.colormaps['viridis']
        pl = a.plot.contourf(cmap=copy(cmap_expected), vmin=0.1, vmax=0.9)
        cmap = pl.cmap
        assert cmap is not None
        assert_array_equal(cmap(np.ma.masked_invalid([np.nan]))[0], cmap_expected(np.ma.masked_invalid([np.nan]))[0])
        assert cmap(-np.inf) == cmap_expected(-np.inf)
        assert cmap(np.inf) == cmap_expected(np.inf)

    def test_contourf_cmap_set_with_bad_under_over(self) -> None:
        a = DataArray(easy_array((4, 4)), dims=['z', 'time'])
        cmap_expected = copy(mpl.colormaps['viridis'])
        cmap_expected.set_bad('w')
        assert np.all(cmap_expected(np.ma.masked_invalid([np.nan]))[0] != mpl.colormaps['viridis'](np.ma.masked_invalid([np.nan]))[0])
        cmap_expected.set_under('r')
        assert cmap_expected(-np.inf) != mpl.colormaps['viridis'](-np.inf)
        cmap_expected.set_over('g')
        assert cmap_expected(np.inf) != mpl.colormaps['viridis'](-np.inf)
        pl = a.plot.contourf(cmap=copy(cmap_expected))
        cmap = pl.cmap
        assert cmap is not None
        assert_array_equal(cmap(np.ma.masked_invalid([np.nan]))[0], cmap_expected(np.ma.masked_invalid([np.nan]))[0])
        assert cmap(-np.inf) == cmap_expected(-np.inf)
        assert cmap(np.inf) == cmap_expected(np.inf)

    def test3d(self) -> None:
        self.darray.plot()

    def test_can_pass_in_axis(self) -> None:
        self.pass_in_axis(self.darray.plot)

    def test__infer_interval_breaks(self) -> None:
        assert_array_equal([-0.5, 0.5, 1.5], _infer_interval_breaks([0, 1]))
        assert_array_equal([-0.5, 0.5, 5.0, 9.5, 10.5], _infer_interval_breaks([0, 1, 9, 10]))
        assert_array_equal(pd.date_range('20000101', periods=4) - np.timedelta64(12, 'h'), _infer_interval_breaks(pd.date_range('20000101', periods=3)))
        xref, yref = np.meshgrid(np.arange(6), np.arange(5))
        cx = (xref[1:, 1:] + xref[:-1, :-1]) / 2
        cy = (yref[1:, 1:] + yref[:-1, :-1]) / 2
        x = _infer_interval_breaks(cx, axis=1)
        x = _infer_interval_breaks(x, axis=0)
        y = _infer_interval_breaks(cy, axis=1)
        y = _infer_interval_breaks(y, axis=0)
        np.testing.assert_allclose(xref, x)
        np.testing.assert_allclose(yref, y)
        with pytest.raises(ValueError):
            _infer_interval_breaks(np.array([0, 2, 1]), check_monotonic=True)

    def test__infer_interval_breaks_logscale(self) -> None:
        """
        Check if interval breaks are defined in the logspace if scale="log"
        """
        x = np.logspace(-4, 3, 8)
        expected_interval_breaks = 10 ** np.linspace(-4.5, 3.5, 9)
        np.testing.assert_allclose(_infer_interval_breaks(x, scale='log'), expected_interval_breaks)
        x = np.logspace(-4, 3, 8)
        y = np.linspace(-5, 5, 11)
        x, y = np.meshgrid(x, y)
        expected_interval_breaks = np.vstack([10 ** np.linspace(-4.5, 3.5, 9)] * 12)
        x = _infer_interval_breaks(x, axis=1, scale='log')
        x = _infer_interval_breaks(x, axis=0, scale='log')
        np.testing.assert_allclose(x, expected_interval_breaks)

    def test__infer_interval_breaks_logscale_invalid_coords(self) -> None:
        """
        Check error is raised when passing non-positive coordinates with logscale
        """
        x = np.linspace(0, 5, 6)
        with pytest.raises(ValueError):
            _infer_interval_breaks(x, scale='log')
        x = np.linspace(-5, 5, 11)
        with pytest.raises(ValueError):
            _infer_interval_breaks(x, scale='log')

    def test_geo_data(self) -> None:
        lat = np.array([[16.28, 18.48, 19.58, 19.54, 18.35], [28.07, 30.52, 31.73, 31.68, 30.37], [39.65, 42.27, 43.56, 43.51, 42.11], [50.52, 53.22, 54.55, 54.5, 53.06]])
        lon = np.array([[-126.13, -113.69, -100.92, -88.04, -75.29], [-129.27, -115.62, -101.54, -87.32, -73.26], [-133.1, -118.0, -102.31, -86.42, -70.76], [-137.85, -120.99, -103.28, -85.28, -67.62]])
        data = np.sqrt(lon ** 2 + lat ** 2)
        da = DataArray(data, dims=('y', 'x'), coords={'lon': (('y', 'x'), lon), 'lat': (('y', 'x'), lat)})
        da.plot(x='lon', y='lat')
        ax = plt.gca()
        assert ax.has_data()
        da.plot(x='lat', y='lon')
        ax = plt.gca()
        assert ax.has_data()

    def test_datetime_dimension(self) -> None:
        nrow = 3
        ncol = 4
        time = pd.date_range('2000-01-01', periods=nrow)
        a = DataArray(easy_array((nrow, ncol)), coords=[('time', time), ('y', range(ncol))])
        a.plot()
        ax = plt.gca()
        assert ax.has_data()

    def test_date_dimension(self) -> None:
        nrow = 3
        ncol = 4
        start = date(2000, 1, 1)
        time = [start + timedelta(days=i) for i in range(nrow)]
        a = DataArray(easy_array((nrow, ncol)), coords=[('time', time), ('y', range(ncol))])
        a.plot()
        ax = plt.gca()
        assert ax.has_data()

    @pytest.mark.slow
    @pytest.mark.filterwarnings('ignore:tight_layout cannot')
    def test_convenient_facetgrid(self) -> None:
        a = easy_array((10, 15, 4))
        d = DataArray(a, dims=['y', 'x', 'z'])
        d.coords['z'] = list('abcd')
        g = d.plot(x='x', y='y', col='z', col_wrap=2, cmap='cool')
        assert_array_equal(g.axs.shape, [2, 2])
        for ax in g.axs.flat:
            assert ax.has_data()
        with pytest.raises(ValueError, match='[Ff]acet'):
            d.plot(x='x', y='y', col='z', ax=plt.gca())
        with pytest.raises(ValueError, match='[Ff]acet'):
            d[0].plot(x='x', y='y', col='z', ax=plt.gca())

    @pytest.mark.slow
    def test_subplot_kws(self) -> None:
        a = easy_array((10, 15, 4))
        d = DataArray(a, dims=['y', 'x', 'z'])
        d.coords['z'] = list('abcd')
        g = d.plot(x='x', y='y', col='z', col_wrap=2, cmap='cool', subplot_kws=dict(facecolor='r'))
        for ax in g.axs.flat:
            assert ax.get_facecolor()[0:3] == mpl.colors.to_rgb('r')

    @pytest.mark.slow
    def test_plot_size(self) -> None:
        self.darray[:, 0, 0].plot(figsize=(13, 5))
        assert tuple(plt.gcf().get_size_inches()) == (13, 5)
        self.darray.plot(figsize=(13, 5))
        assert tuple(plt.gcf().get_size_inches()) == (13, 5)
        self.darray.plot(size=5)
        assert plt.gcf().get_size_inches()[1] == 5
        self.darray.plot(size=5, aspect=2)
        assert tuple(plt.gcf().get_size_inches()) == (10, 5)
        with pytest.raises(ValueError, match='cannot provide both'):
            self.darray.plot(ax=plt.gca(), figsize=(3, 4))
        with pytest.raises(ValueError, match='cannot provide both'):
            self.darray.plot(size=5, figsize=(3, 4))
        with pytest.raises(ValueError, match='cannot provide both'):
            self.darray.plot(size=5, ax=plt.gca())
        with pytest.raises(ValueError, match='cannot provide `aspect`'):
            self.darray.plot(aspect=1)

    @pytest.mark.slow
    @pytest.mark.filterwarnings('ignore:tight_layout cannot')
    def test_convenient_facetgrid_4d(self) -> None:
        a = easy_array((10, 15, 2, 3))
        d = DataArray(a, dims=['y', 'x', 'columns', 'rows'])
        g = d.plot(x='x', y='y', col='columns', row='rows')
        assert_array_equal(g.axs.shape, [3, 2])
        for ax in g.axs.flat:
            assert ax.has_data()
        with pytest.raises(ValueError, match='[Ff]acet'):
            d.plot(x='x', y='y', col='columns', ax=plt.gca())

    def test_coord_with_interval(self) -> None:
        """Test line plot with intervals."""
        bins = [-1, 0, 1, 2]
        self.darray.groupby_bins('dim_0', bins).mean(...).plot()

    def test_coord_with_interval_x(self) -> None:
        """Test line plot with intervals explicitly on x axis."""
        bins = [-1, 0, 1, 2]
        self.darray.groupby_bins('dim_0', bins).mean(...).plot(x='dim_0_bins')

    def test_coord_with_interval_y(self) -> None:
        """Test line plot with intervals explicitly on y axis."""
        bins = [-1, 0, 1, 2]
        self.darray.groupby_bins('dim_0', bins).mean(...).plot(y='dim_0_bins')

    def test_coord_with_interval_xy(self) -> None:
        """Test line plot with intervals on both x and y axes."""
        bins = [-1, 0, 1, 2]
        self.darray.groupby_bins('dim_0', bins).mean(...).dim_0_bins.plot()

    @pytest.mark.parametrize('dim', ('x', 'y'))
    def test_labels_with_units_with_interval(self, dim) -> None:
        """Test line plot with intervals and a units attribute."""
        bins = [-1, 0, 1, 2]
        arr = self.darray.groupby_bins('dim_0', bins).mean(...)
        arr.dim_0_bins.attrs['units'] = 'm'
        mappable, = arr.plot(**{dim: 'dim_0_bins'})
        ax = mappable.figure.gca()
        actual = getattr(ax, f'get_{dim}label')()
        expected = 'dim_0_bins_center [m]'
        assert actual == expected

    def test_multiplot_over_length_one_dim(self) -> None:
        a = easy_array((3, 1, 1, 1))
        d = DataArray(a, dims=('x', 'col', 'row', 'hue'))
        d.plot(col='col')
        d.plot(row='row')
        d.plot(hue='hue')