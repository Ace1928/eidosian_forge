from __future__ import annotations
import functools
import operator
import numpy as np
import pytest
import xarray as xr
from xarray.core import dtypes, duck_array_ops
from xarray.tests import (
from xarray.tests.test_plot import PlotTestCase
from xarray.tests.test_variable import _PAD_XR_NP_ARGS
@requires_matplotlib
class TestPlots(PlotTestCase):

    @pytest.mark.parametrize('coord_unit, coord_attrs', [(1, {'units': 'meter'}), pytest.param(unit_registry.m, {}, marks=pytest.mark.xfail(reason="indexes don't support units"))])
    def test_units_in_line_plot_labels(self, coord_unit, coord_attrs):
        arr = np.linspace(1, 10, 3) * unit_registry.Pa
        coord_arr = np.linspace(1, 3, 3) * coord_unit
        x_coord = xr.DataArray(coord_arr, dims='x', attrs=coord_attrs)
        da = xr.DataArray(data=arr, dims='x', coords={'x': x_coord}, name='pressure')
        da.plot.line()
        ax = plt.gca()
        assert ax.get_ylabel() == 'pressure [pascal]'
        assert ax.get_xlabel() == 'x [meter]'

    @pytest.mark.parametrize('coord_unit, coord_attrs', [(1, {'units': 'meter'}), pytest.param(unit_registry.m, {}, marks=pytest.mark.xfail(reason="indexes don't support units"))])
    def test_units_in_slice_line_plot_labels_sel(self, coord_unit, coord_attrs):
        arr = xr.DataArray(name='var_a', data=np.array([[1, 2], [3, 4]]), coords=dict(a=('a', np.array([5, 6]) * coord_unit, coord_attrs), b=('b', np.array([7, 8]) * coord_unit, coord_attrs)), dims=('a', 'b'))
        arr.sel(a=5).plot(marker='o')
        assert plt.gca().get_title() == 'a = 5 [meter]'

    @pytest.mark.parametrize('coord_unit, coord_attrs', [(1, {'units': 'meter'}), pytest.param(unit_registry.m, {}, marks=pytest.mark.xfail(reason='pint.errors.UnitStrippedWarning'))])
    def test_units_in_slice_line_plot_labels_isel(self, coord_unit, coord_attrs):
        arr = xr.DataArray(name='var_a', data=np.array([[1, 2], [3, 4]]), coords=dict(a=('x', np.array([5, 6]) * coord_unit, coord_attrs), b=('y', np.array([7, 8]))), dims=('x', 'y'))
        arr.isel(x=0).plot(marker='o')
        assert plt.gca().get_title() == 'a = 5 [meter]'

    def test_units_in_2d_plot_colorbar_label(self):
        arr = np.ones((2, 3)) * unit_registry.Pa
        da = xr.DataArray(data=arr, dims=['x', 'y'], name='pressure')
        fig, (ax, cax) = plt.subplots(1, 2)
        ax = da.plot.contourf(ax=ax, cbar_ax=cax, add_colorbar=True)
        assert cax.get_ylabel() == 'pressure [pascal]'

    def test_units_facetgrid_plot_labels(self):
        arr = np.ones((2, 3)) * unit_registry.Pa
        da = xr.DataArray(data=arr, dims=['x', 'y'], name='pressure')
        fig, (ax, cax) = plt.subplots(1, 2)
        fgrid = da.plot.line(x='x', col='y')
        assert fgrid.axs[0, 0].get_ylabel() == 'pressure [pascal]'

    def test_units_facetgrid_2d_imshow_plot_colorbar_labels(self):
        arr = np.ones((2, 3, 4, 5)) * unit_registry.Pa
        da = xr.DataArray(data=arr, dims=['x', 'y', 'z', 'w'], name='pressure')
        da.plot.imshow(x='x', y='y', col='w')

    def test_units_facetgrid_2d_contourf_plot_colorbar_labels(self):
        arr = np.ones((2, 3, 4)) * unit_registry.Pa
        da = xr.DataArray(data=arr, dims=['x', 'y', 'z'], name='pressure')
        fig, (ax1, ax2, ax3, cax) = plt.subplots(1, 4)
        fgrid = da.plot.contourf(x='x', y='y', col='z')
        assert fgrid.cbar.ax.get_ylabel() == 'pressure [pascal]'