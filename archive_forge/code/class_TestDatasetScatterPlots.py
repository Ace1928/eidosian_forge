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
class TestDatasetScatterPlots(PlotTestCase):

    @pytest.fixture(autouse=True)
    def setUp(self) -> None:
        das = [DataArray(np.random.randn(3, 3, 4, 4), dims=['x', 'row', 'col', 'hue'], coords=[range(k) for k in [3, 3, 4, 4]]) for _ in [1, 2]]
        ds = Dataset({'A': das[0], 'B': das[1]})
        ds.hue.name = 'huename'
        ds.hue.attrs['units'] = 'hunits'
        ds.x.attrs['units'] = 'xunits'
        ds.col.attrs['units'] = 'colunits'
        ds.row.attrs['units'] = 'rowunits'
        ds.A.attrs['units'] = 'Aunits'
        ds.B.attrs['units'] = 'Bunits'
        self.ds = ds

    def test_accessor(self) -> None:
        from xarray.plot.accessor import DatasetPlotAccessor
        assert Dataset.plot is DatasetPlotAccessor
        assert isinstance(self.ds.plot, DatasetPlotAccessor)

    @pytest.mark.parametrize('add_guide, hue_style, legend, colorbar', [(None, None, False, True), (False, None, False, False), (True, None, False, True), (True, 'continuous', False, True), (False, 'discrete', False, False), (True, 'discrete', True, False)])
    def test_add_guide(self, add_guide: bool | None, hue_style: Literal['continuous', 'discrete', None], legend: bool, colorbar: bool) -> None:
        meta_data = _infer_meta_data(self.ds, x='A', y='B', hue='hue', hue_style=hue_style, add_guide=add_guide, funcname='scatter')
        assert meta_data['add_legend'] is legend
        assert meta_data['add_colorbar'] is colorbar

    def test_facetgrid_shape(self) -> None:
        g = self.ds.plot.scatter(x='A', y='B', row='row', col='col')
        assert g.axs.shape == (len(self.ds.row), len(self.ds.col))
        g = self.ds.plot.scatter(x='A', y='B', row='col', col='row')
        assert g.axs.shape == (len(self.ds.col), len(self.ds.row))

    def test_default_labels(self) -> None:
        g = self.ds.plot.scatter(x='A', y='B', row='row', col='col', hue='hue')
        for label, ax in zip(self.ds.coords['col'].values, g.axs[0, :]):
            assert substring_in_axes(str(label), ax)
        for ax in g.axs[-1, :]:
            assert ax.get_xlabel() == 'A [Aunits]'
        for ax in g.axs[:, 0]:
            assert ax.get_ylabel() == 'B [Bunits]'

    def test_axes_in_faceted_plot(self) -> None:
        with pytest.raises(ValueError):
            self.ds.plot.scatter(x='A', y='B', row='row', ax=plt.axes())

    def test_figsize_and_size(self) -> None:
        with pytest.raises(ValueError):
            self.ds.plot.scatter(x='A', y='B', row='row', size=3, figsize=(4, 3))

    @pytest.mark.parametrize('x, y, hue, add_legend, add_colorbar, error_type', [pytest.param('A', 'The Spanish Inquisition', None, None, None, KeyError, id='bad_y'), pytest.param('The Spanish Inquisition', 'B', None, None, True, ValueError, id='bad_x')])
    def test_bad_args(self, x: Hashable, y: Hashable, hue: Hashable | None, add_legend: bool | None, add_colorbar: bool | None, error_type: type[Exception]):
        with pytest.raises(error_type):
            self.ds.plot.scatter(x=x, y=y, hue=hue, add_legend=add_legend, add_colorbar=add_colorbar)

    def test_datetime_hue(self) -> None:
        ds2 = self.ds.copy()
        ds2['hue'] = pd.date_range('2000-1-1', periods=4)
        ds2.plot.scatter(x='A', y='B', hue='hue')
        ds2['hue'] = pd.timedelta_range('-1D', periods=4, freq='D')
        ds2.plot.scatter(x='A', y='B', hue='hue')

    def test_facetgrid_hue_style(self) -> None:
        ds2 = self.ds.copy()
        g = ds2.plot.scatter(x='A', y='B', row='row', col='col', hue='hue')
        assert isinstance(g._mappables[-1], mpl.collections.PathCollection)
        ds2['hue'] = pd.date_range('2000-1-1', periods=4)
        g = ds2.plot.scatter(x='A', y='B', row='row', col='col', hue='hue')
        assert isinstance(g._mappables[-1], mpl.collections.PathCollection)
        ds2['hue'] = ['a', 'a', 'b', 'b']
        g = ds2.plot.scatter(x='A', y='B', row='row', col='col', hue='hue')
        assert isinstance(g._mappables[-1], mpl.collections.PathCollection)

    @pytest.mark.parametrize(['x', 'y', 'hue', 'markersize'], [('A', 'B', 'x', 'col'), ('x', 'row', 'A', 'B')])
    def test_scatter(self, x: Hashable, y: Hashable, hue: Hashable, markersize: Hashable) -> None:
        self.ds.plot.scatter(x=x, y=y, hue=hue, markersize=markersize)
        with pytest.raises(ValueError, match='u, v'):
            self.ds.plot.scatter(x=x, y=y, u='col', v='row')

    def test_non_numeric_legend(self) -> None:
        ds2 = self.ds.copy()
        ds2['hue'] = ['a', 'b', 'c', 'd']
        pc = ds2.plot.scatter(x='A', y='B', markersize='hue')
        axes = pc.axes
        assert axes is not None
        assert hasattr(axes, 'legend_')
        assert axes.legend_ is not None

    def test_legend_labels(self) -> None:
        ds2 = self.ds.copy()
        ds2['hue'] = ['a', 'a', 'b', 'b']
        pc = ds2.plot.scatter(x='A', y='B', markersize='hue')
        axes = pc.axes
        assert axes is not None
        actual = [t.get_text() for t in axes.get_legend().texts]
        expected = ['hue', 'a', 'b']
        assert actual == expected

    def test_legend_labels_facetgrid(self) -> None:
        ds2 = self.ds.copy()
        ds2['hue'] = ['d', 'a', 'c', 'b']
        g = ds2.plot.scatter(x='A', y='B', hue='hue', markersize='x', col='col')
        legend = g.figlegend
        assert legend is not None
        actual = tuple((t.get_text() for t in legend.texts))
        expected = ('x [xunits]', '$\\mathdefault{0}$', '$\\mathdefault{1}$', '$\\mathdefault{2}$')
        assert actual == expected

    def test_add_legend_by_default(self) -> None:
        sc = self.ds.plot.scatter(x='A', y='B', hue='hue')
        fig = sc.figure
        assert fig is not None
        assert len(fig.axes) == 2