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
class TestLineStringPlotting:

    def setup_method(self):
        self.N = 10
        values = np.arange(self.N)
        self.lines = GeoSeries([LineString([(0, i), (4, i + 0.5), (9, i)]) for i in range(self.N)], index=list('ABCDEFGHIJ'))
        self.df = GeoDataFrame({'geometry': self.lines, 'values': values})
        multiline1 = MultiLineString(self.lines.loc['A':'B'].values)
        multiline2 = MultiLineString(self.lines.loc['C':'D'].values)
        self.df2 = GeoDataFrame({'geometry': [multiline1, multiline2], 'values': [0, 1]})
        self.linearrings = GeoSeries([LinearRing([(0, i), (4, i + 0.5), (9, i)]) for i in range(self.N)], index=list('ABCDEFGHIJ'))
        self.df3 = GeoDataFrame({'geometry': self.linearrings, 'values': values})

    def test_single_color(self):
        ax = self.lines.plot(color='green')
        _check_colors(self.N, ax.collections[0].get_colors(), ['green'] * self.N)
        ax = self.df.plot(color='green')
        _check_colors(self.N, ax.collections[0].get_colors(), ['green'] * self.N)
        ax = self.linearrings.plot(color='green')
        _check_colors(self.N, ax.collections[0].get_colors(), ['green'] * self.N)
        ax = self.df3.plot(color='green')
        _check_colors(self.N, ax.collections[0].get_colors(), ['green'] * self.N)
        ax = self.df.plot(color=(0.5, 0.5, 0.5, 0.5))
        _check_colors(self.N, ax.collections[0].get_colors(), [(0.5, 0.5, 0.5, 0.5)] * self.N)
        ax = self.df.plot(color=(0.5, 0.5, 0.5, 0.5))
        _check_colors(self.N, ax.collections[0].get_colors(), [(0.5, 0.5, 0.5, 0.5)] * self.N)
        with pytest.raises((TypeError, ValueError)):
            self.df.plot(color='not color')
        with warnings.catch_warnings(record=True) as _:
            ax = self.df.plot(column='values', color='green')
            _check_colors(self.N, ax.collections[0].get_colors(), ['green'] * self.N)

    def test_style_kwargs_linestyle(self):
        for ax in [self.lines.plot(linestyle=':', linewidth=1), self.df.plot(linestyle=':', linewidth=1), self.df.plot(column='values', linestyle=':', linewidth=1)]:
            assert [(0.0, [1.0, 1.65])] == ax.collections[0].get_linestyle()
        ax = self.lines.plot(linestyle=(0, (3, 10, 1, 15)), linewidth=1)
        assert [(0, [3, 10, 1, 15])] == ax.collections[0].get_linestyle()
        ls = [('dashed', 'dotted', 'dashdot', 'solid')[k % 4] for k in range(self.N)]
        exp_ls = [_style_to_linestring_onoffseq(st, 1) for st in ls]
        for ax in [self.lines.plot(linestyle=ls, linewidth=1), self.lines.plot(linestyles=ls, linewidth=1), self.df.plot(linestyle=ls, linewidth=1), self.df.plot(column='values', linestyle=ls, linewidth=1)]:
            assert exp_ls == ax.collections[0].get_linestyle()

    def test_style_kwargs_linewidth(self):
        for ax in [self.lines.plot(linewidth=2), self.df.plot(linewidth=2), self.df.plot(column='values', linewidth=2)]:
            np.testing.assert_array_equal([2], ax.collections[0].get_linewidths())
        lw = [(0, 1, 2, 5.5, 10)[k % 5] for k in range(self.N)]
        for ax in [self.lines.plot(linewidth=lw), self.lines.plot(linewidths=lw), self.df.plot(linewidth=lw), self.df.plot(column='values', linewidth=lw)]:
            np.testing.assert_array_equal(lw, ax.collections[0].get_linewidths())

    def test_style_kwargs_alpha(self):
        ax = self.df.plot(alpha=0.7)
        np.testing.assert_array_equal([0.7], ax.collections[0].get_alpha())
        try:
            ax = self.df.plot(alpha=np.linspace(0, 0.0, 1.0, self.N))
        except TypeError:
            pass
        else:
            np.testing.assert_array_equal(np.linspace(0, 0.0, 1.0, self.N), ax.collections[0].get_alpha())

    def test_style_kwargs_path_effects(self):
        from matplotlib.patheffects import withStroke
        effects = [withStroke(linewidth=8, foreground='b')]
        ax = self.df.plot(color='orange', path_effects=effects)
        assert ax.collections[0].get_path_effects()[0].__dict__['_gc'] == {'linewidth': 8, 'foreground': 'b'}

    def test_subplots_norm(self):
        cmap = matplotlib.cm.viridis_r
        norm = matplotlib.colors.Normalize(vmin=0, vmax=20)
        ax = self.df.plot(column='values', cmap=cmap, norm=norm)
        actual_colors_orig = ax.collections[0].get_edgecolors()
        exp_colors = cmap(np.arange(10) / 20)
        np.testing.assert_array_equal(exp_colors, actual_colors_orig)
        fig, ax = plt.subplots()
        self.df[1:].plot(column='values', ax=ax, norm=norm, cmap=cmap)
        actual_colors_sub = ax.collections[0].get_edgecolors()
        np.testing.assert_array_equal(actual_colors_orig[1], actual_colors_sub[0])

    def test_multilinestrings(self):
        ax = self.df2.plot()
        assert len(ax.collections[0].get_paths()) == 4
        _check_colors(4, ax.collections[0].get_edgecolors(), [MPL_DFT_COLOR] * 4)
        ax = self.df2.plot('values')
        cmap = plt.get_cmap(lut=2)
        expected_colors = [cmap(0), cmap(0), cmap(1), cmap(1)]
        _check_colors(4, ax.collections[0].get_edgecolors(), expected_colors)
        ax = self.df2.plot(color=['r', 'b'])
        _check_colors(4, ax.collections[0].get_edgecolors(), ['r', 'r', 'b', 'b'])