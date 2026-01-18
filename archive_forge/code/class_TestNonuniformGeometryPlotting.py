import itertools
from packaging.version import Version
import warnings
import numpy as np
import pandas as pd
from shapely import wkt
from shapely.affinity import rotate
from shapely.geometry import (
from geopandas import GeoDataFrame, GeoSeries, read_file
from geopandas.datasets import get_path
import geopandas._compat as compat
from geopandas.plotting import GeoplotAccessor
import pytest
import matplotlib.pyplot as plt
class TestNonuniformGeometryPlotting:

    def setup_method(self):
        pytest.importorskip('matplotlib')
        poly = Polygon([(1, 0), (2, 0), (2, 1)])
        line = LineString([(0.5, 0.5), (1, 1), (1, 0.5), (1.5, 1)])
        point = Point(0.75, 0.25)
        self.series = GeoSeries([poly, line, point])
        self.df = GeoDataFrame({'geometry': self.series, 'values': [1, 2, 3]})

    def test_colors(self):
        ax = self.series.plot()
        _check_colors(1, ax.collections[0].get_facecolors(), [MPL_DFT_COLOR])
        _check_colors(1, ax.collections[1].get_edgecolors(), [MPL_DFT_COLOR])
        _check_colors(1, ax.collections[2].get_facecolors(), [MPL_DFT_COLOR])
        ax = self.series.plot(cmap='RdYlGn')
        cmap = plt.get_cmap('RdYlGn')
        exp_colors = cmap(np.arange(3) / (3 - 1))
        _check_colors(1, ax.collections[0].get_facecolors(), [exp_colors[0]])
        _check_colors(1, ax.collections[1].get_edgecolors(), [exp_colors[1]])
        _check_colors(1, ax.collections[2].get_facecolors(), [exp_colors[2]])

    def test_style_kwargs(self):
        ax = self.series.plot(markersize=10)
        assert ax.collections[2].get_sizes() == [10]
        ax = self.df.plot(markersize=10)
        assert ax.collections[2].get_sizes() == [10]

    def test_style_kwargs_linestyle(self):
        for ax in [self.series.plot(linestyle=':', linewidth=1), self.df.plot(linestyle=':', linewidth=1)]:
            assert [(0.0, [1.0, 1.65])] == ax.collections[0].get_linestyle()
        ax = self.series.plot(linestyle=(0, (3, 10, 1, 15)), linewidth=1)
        assert [(0, [3, 10, 1, 15])] == ax.collections[0].get_linestyle()

    @pytest.mark.skip(reason='array-like style_kwds not supported for mixed geometry types (#1379)')
    def test_style_kwargs_linestyle_listlike(self):
        ls = ['solid', 'dotted', 'dashdot']
        exp_ls = [_style_to_linestring_onoffseq(style, 1) for style in ls]
        for ax in [self.series.plot(linestyle=ls, linewidth=1), self.series.plot(linestyles=ls, linewidth=1), self.df.plot(linestyles=ls, linewidth=1)]:
            assert exp_ls == ax.collections[0].get_linestyle()

    def test_style_kwargs_linewidth(self):
        ax = self.df.plot(linewidth=2)
        np.testing.assert_array_equal([2], ax.collections[0].get_linewidths())

    @pytest.mark.skip(reason='array-like style_kwds not supported for mixed geometry types (#1379)')
    def test_style_kwargs_linewidth_listlike(self):
        for ax in [self.series.plot(linewidths=[2, 4, 5.5]), self.series.plot(linewidths=[2, 4, 5.5]), self.df.plot(linewidths=[2, 4, 5.5])]:
            np.testing.assert_array_equal([2, 4, 5.5], ax.collections[0].get_linewidths())

    def test_style_kwargs_alpha(self):
        ax = self.df.plot(alpha=0.7)
        np.testing.assert_array_equal([0.7], ax.collections[0].get_alpha())