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
class TestPolygonPlotting:

    def setup_method(self):
        t1 = Polygon([(0, 0), (1, 0), (1, 1)])
        t2 = Polygon([(1, 0), (2, 0), (2, 1)])
        self.polys = GeoSeries([t1, t2], index=list('AB'))
        self.df = GeoDataFrame({'geometry': self.polys, 'values': [0, 1]})
        multipoly1 = MultiPolygon([t1, t2])
        multipoly2 = rotate(multipoly1, 180)
        self.df2 = GeoDataFrame({'geometry': [multipoly1, multipoly2], 'values': [0, 1]})
        t3 = Polygon([(2, 0), (3, 0), (3, 1)])
        df_nan = GeoDataFrame({'geometry': t3, 'values': [np.nan]})
        self.df3 = pd.concat([self.df, df_nan])

    def test_single_color(self):
        ax = self.polys.plot(color='green')
        _check_colors(2, ax.collections[0].get_facecolors(), ['green'] * 2)
        assert len(ax.collections[0].get_edgecolors()) == 0
        ax = self.df.plot(color='green')
        _check_colors(2, ax.collections[0].get_facecolors(), ['green'] * 2)
        assert len(ax.collections[0].get_edgecolors()) == 0
        ax = self.df.plot(color=(0.5, 0.5, 0.5))
        _check_colors(2, ax.collections[0].get_facecolors(), [(0.5, 0.5, 0.5)] * 2)
        ax = self.df.plot(color=(0.5, 0.5, 0.5, 0.5))
        _check_colors(2, ax.collections[0].get_facecolors(), [(0.5, 0.5, 0.5, 0.5)] * 2)
        with pytest.raises((TypeError, ValueError)):
            self.df.plot(color='not color')
        with warnings.catch_warnings(record=True) as _:
            ax = self.df.plot(column='values', color='green')
            _check_colors(2, ax.collections[0].get_facecolors(), ['green'] * 2)

    def test_vmin_vmax(self):
        ax = self.df.plot(column='values', categorical=False, vmin=0, vmax=0)
        actual_colors = ax.collections[0].get_facecolors()
        np.testing.assert_array_equal(actual_colors[0], actual_colors[1])
        ax = self.df.plot(column='values', categorical=True, vmin=0, vmax=0)
        actual_colors = ax.collections[0].get_facecolors()
        np.testing.assert_array_equal(actual_colors[0], actual_colors[1])
        ax = self.df3.plot(column='values')
        actual_colors = ax.collections[0].get_facecolors()
        assert np.any(np.not_equal(actual_colors[0], actual_colors[1]))

    def test_style_kwargs_color(self):
        ax = self.polys.plot(facecolor='k')
        _check_colors(2, ax.collections[0].get_facecolors(), ['k'] * 2)
        ax = self.polys.plot(color='red', facecolor='k')
        ax = self.polys.plot(edgecolor='red')
        np.testing.assert_array_equal([(1, 0, 0, 1)], ax.collections[0].get_edgecolors())
        ax = self.df.plot('values', edgecolor='red')
        np.testing.assert_array_equal([(1, 0, 0, 1)], ax.collections[0].get_edgecolors())
        ax = self.polys.plot(facecolor='g', edgecolor='r', alpha=0.4)
        _check_colors(2, ax.collections[0].get_facecolors(), ['g'] * 2, alpha=0.4)
        _check_colors(2, ax.collections[0].get_edgecolors(), ['r'] * 2, alpha=0.4)
        ax = self.df.plot(facecolor=(0.5, 0.5, 0.5), edgecolor=(0.4, 0.5, 0.6))
        _check_colors(2, ax.collections[0].get_facecolors(), [(0.5, 0.5, 0.5)] * 2)
        _check_colors(2, ax.collections[0].get_edgecolors(), [(0.4, 0.5, 0.6)] * 2)
        ax = self.df.plot(facecolor=(0.5, 0.5, 0.5, 0.5), edgecolor=(0.4, 0.5, 0.6, 0.5))
        _check_colors(2, ax.collections[0].get_facecolors(), [(0.5, 0.5, 0.5, 0.5)] * 2)
        _check_colors(2, ax.collections[0].get_edgecolors(), [(0.4, 0.5, 0.6, 0.5)] * 2)

    def test_style_kwargs_linestyle(self):
        ax = self.df.plot(linestyle=':', linewidth=1)
        assert [(0.0, [1.0, 1.65])] == ax.collections[0].get_linestyle()
        ax = self.df.plot(linestyle=(0, (3, 10, 1, 15)), linewidth=1)
        assert [(0, [3, 10, 1, 15])] == ax.collections[0].get_linestyle()
        ls = ['dashed', 'dotted']
        exp_ls = [_style_to_linestring_onoffseq(st, 1) for st in ls]
        for ax in [self.df.plot(linestyle=ls, linewidth=1), self.df.plot(linestyles=ls, linewidth=1)]:
            assert exp_ls == ax.collections[0].get_linestyle()

    def test_style_kwargs_linewidth(self):
        ax = self.df.plot(linewidth=2)
        np.testing.assert_array_equal([2], ax.collections[0].get_linewidths())
        for ax in [self.df.plot(linewidth=[2, 4]), self.df.plot(linewidths=[2, 4])]:
            np.testing.assert_array_equal([2, 4], ax.collections[0].get_linewidths())
        ax = self.df.plot(alpha=0.7)
        np.testing.assert_array_equal([0.7], ax.collections[0].get_alpha())
        try:
            ax = self.df.plot(alpha=[0.7, 0.2])
        except TypeError:
            pass
        else:
            np.testing.assert_array_equal([0.7, 0.2], ax.collections[0].get_alpha())

    def test_legend_kwargs(self):
        categories = list(self.df['values'].unique())
        prefix = 'LABEL_FOR_'
        ax = self.df.plot(column='values', categorical=True, categories=categories, legend=True, legend_kwds={'labels': [prefix + str(c) for c in categories], 'frameon': False})
        assert len(categories) == len(ax.get_legend().get_texts())
        assert ax.get_legend().get_texts()[0].get_text().startswith(prefix)
        assert ax.get_legend().get_frame_on() is False

    def test_colorbar_kwargs(self):
        label_txt = 'colorbar test'
        ax = self.df.plot(column='values', categorical=False, legend=True, legend_kwds={'label': label_txt})
        cax = _get_colorbar_ax(ax.get_figure())
        assert cax.get_ylabel() == label_txt
        ax = self.df.plot(column='values', categorical=False, legend=True, legend_kwds={'label': label_txt, 'orientation': 'horizontal'})
        cax = _get_colorbar_ax(ax.get_figure())
        assert cax.get_xlabel() == label_txt

    def test_fmt_ignore(self):
        self.df.plot(column='values', categorical=True, legend=True, legend_kwds={'fmt': '{:.0f}'})
        self.df.plot(column='values', legend=True, legend_kwds={'fmt': '{:.0f}'})

    def test_multipolygons_color(self):
        ax = self.df2.plot()
        assert len(ax.collections[0].get_paths()) == 4
        _check_colors(4, ax.collections[0].get_facecolors(), [MPL_DFT_COLOR] * 4)
        ax = self.df2.plot('values')
        cmap = plt.get_cmap(lut=2)
        expected_colors = [cmap(0), cmap(0), cmap(1), cmap(1)]
        _check_colors(4, ax.collections[0].get_facecolors(), expected_colors)
        ax = self.df2.plot(color=['r', 'b'])
        _check_colors(4, ax.collections[0].get_facecolors(), ['r', 'r', 'b', 'b'])

    def test_multipolygons_linestyle(self):
        ax = self.df2.plot(linestyle=':', linewidth=1)
        assert [(0.0, [1.0, 1.65])] == ax.collections[0].get_linestyle()
        ax = self.df2.plot(linestyle=(0, (3, 10, 1, 15)), linewidth=1)
        assert [(0, [3, 10, 1, 15])] == ax.collections[0].get_linestyle()
        ls = ['dashed', 'dotted']
        exp_ls = [_style_to_linestring_onoffseq(st, 1) for st in ls for i in range(2)]
        for ax in [self.df2.plot(linestyle=ls, linewidth=1), self.df2.plot(linestyles=ls, linewidth=1)]:
            assert exp_ls == ax.collections[0].get_linestyle()

    def test_multipolygons_linewidth(self):
        ax = self.df2.plot(linewidth=2)
        np.testing.assert_array_equal([2], ax.collections[0].get_linewidths())
        for ax in [self.df2.plot(linewidth=[2, 4]), self.df2.plot(linewidths=[2, 4])]:
            np.testing.assert_array_equal([2, 2, 4, 4], ax.collections[0].get_linewidths())

    def test_multipolygons_alpha(self):
        ax = self.df2.plot(alpha=0.7)
        np.testing.assert_array_equal([0.7], ax.collections[0].get_alpha())
        try:
            ax = self.df2.plot(alpha=[0.7, 0.2])
        except TypeError:
            pass
        else:
            np.testing.assert_array_equal([0.7, 0.7, 0.2, 0.2], ax.collections[0].get_alpha())

    def test_subplots_norm(self):
        cmap = matplotlib.cm.viridis_r
        norm = matplotlib.colors.Normalize(vmin=0, vmax=10)
        ax = self.df.plot(column='values', cmap=cmap, norm=norm)
        actual_colors_orig = ax.collections[0].get_facecolors()
        exp_colors = cmap(np.arange(2) / 10)
        np.testing.assert_array_equal(exp_colors, actual_colors_orig)
        fig, ax = plt.subplots()
        self.df[1:].plot(column='values', ax=ax, norm=norm, cmap=cmap)
        actual_colors_sub = ax.collections[0].get_facecolors()
        np.testing.assert_array_equal(actual_colors_orig[1], actual_colors_sub[0])