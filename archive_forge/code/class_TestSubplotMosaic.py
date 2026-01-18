import copy
from datetime import datetime
import io
from pathlib import Path
import pickle
import platform
from threading import Timer
from types import SimpleNamespace
import warnings
import numpy as np
import pytest
from PIL import Image
import matplotlib as mpl
from matplotlib import gridspec
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib.axes import Axes
from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.figure import Figure, FigureBase
from matplotlib.layout_engine import (ConstrainedLayoutEngine,
from matplotlib.ticker import AutoMinorLocator, FixedFormatter, ScalarFormatter
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
class TestSubplotMosaic:

    @check_figures_equal(extensions=['png'])
    @pytest.mark.parametrize('x', [[['A', 'A', 'B'], ['C', 'D', 'B']], [[1, 1, 2], [3, 4, 2]], (('A', 'A', 'B'), ('C', 'D', 'B')), ((1, 1, 2), (3, 4, 2))])
    def test_basic(self, fig_test, fig_ref, x):
        grid_axes = fig_test.subplot_mosaic(x)
        for k, ax in grid_axes.items():
            ax.set_title(k)
        labels = sorted(np.unique(x))
        assert len(labels) == len(grid_axes)
        gs = fig_ref.add_gridspec(2, 3)
        axA = fig_ref.add_subplot(gs[:1, :2])
        axA.set_title(labels[0])
        axB = fig_ref.add_subplot(gs[:, 2])
        axB.set_title(labels[1])
        axC = fig_ref.add_subplot(gs[1, 0])
        axC.set_title(labels[2])
        axD = fig_ref.add_subplot(gs[1, 1])
        axD.set_title(labels[3])

    @check_figures_equal(extensions=['png'])
    def test_all_nested(self, fig_test, fig_ref):
        x = [['A', 'B'], ['C', 'D']]
        y = [['E', 'F'], ['G', 'H']]
        fig_ref.set_layout_engine('constrained')
        fig_test.set_layout_engine('constrained')
        grid_axes = fig_test.subplot_mosaic([[x, y]])
        for ax in grid_axes.values():
            ax.set_title(ax.get_label())
        gs = fig_ref.add_gridspec(1, 2)
        gs_left = gs[0, 0].subgridspec(2, 2)
        for j, r in enumerate(x):
            for k, label in enumerate(r):
                fig_ref.add_subplot(gs_left[j, k]).set_title(label)
        gs_right = gs[0, 1].subgridspec(2, 2)
        for j, r in enumerate(y):
            for k, label in enumerate(r):
                fig_ref.add_subplot(gs_right[j, k]).set_title(label)

    @check_figures_equal(extensions=['png'])
    def test_nested(self, fig_test, fig_ref):
        fig_ref.set_layout_engine('constrained')
        fig_test.set_layout_engine('constrained')
        x = [['A', 'B'], ['C', 'D']]
        y = [['F'], [x]]
        grid_axes = fig_test.subplot_mosaic(y)
        for k, ax in grid_axes.items():
            ax.set_title(k)
        gs = fig_ref.add_gridspec(2, 1)
        gs_n = gs[1, 0].subgridspec(2, 2)
        axA = fig_ref.add_subplot(gs_n[0, 0])
        axA.set_title('A')
        axB = fig_ref.add_subplot(gs_n[0, 1])
        axB.set_title('B')
        axC = fig_ref.add_subplot(gs_n[1, 0])
        axC.set_title('C')
        axD = fig_ref.add_subplot(gs_n[1, 1])
        axD.set_title('D')
        axF = fig_ref.add_subplot(gs[0, 0])
        axF.set_title('F')

    @check_figures_equal(extensions=['png'])
    def test_nested_tuple(self, fig_test, fig_ref):
        x = [['A', 'B', 'B'], ['C', 'C', 'D']]
        xt = (('A', 'B', 'B'), ('C', 'C', 'D'))
        fig_ref.subplot_mosaic([['F'], [x]])
        fig_test.subplot_mosaic([['F'], [xt]])

    def test_nested_width_ratios(self):
        x = [['A', [['B'], ['C']]]]
        width_ratios = [2, 1]
        fig, axd = plt.subplot_mosaic(x, width_ratios=width_ratios)
        assert axd['A'].get_gridspec().get_width_ratios() == width_ratios
        assert axd['B'].get_gridspec().get_width_ratios() != width_ratios

    def test_nested_height_ratios(self):
        x = [['A', [['B'], ['C']]], ['D', 'D']]
        height_ratios = [1, 2]
        fig, axd = plt.subplot_mosaic(x, height_ratios=height_ratios)
        assert axd['D'].get_gridspec().get_height_ratios() == height_ratios
        assert axd['B'].get_gridspec().get_height_ratios() != height_ratios

    @check_figures_equal(extensions=['png'])
    @pytest.mark.parametrize('x, empty_sentinel', [([['A', None], [None, 'B']], None), ([['A', '.'], ['.', 'B']], 'SKIP'), ([['A', 0], [0, 'B']], 0), ([[1, None], [None, 2]], None), ([[1, '.'], ['.', 2]], 'SKIP'), ([[1, 0], [0, 2]], 0)])
    def test_empty(self, fig_test, fig_ref, x, empty_sentinel):
        if empty_sentinel != 'SKIP':
            kwargs = {'empty_sentinel': empty_sentinel}
        else:
            kwargs = {}
        grid_axes = fig_test.subplot_mosaic(x, **kwargs)
        for k, ax in grid_axes.items():
            ax.set_title(k)
        labels = sorted({name for row in x for name in row} - {empty_sentinel, '.'})
        assert len(labels) == len(grid_axes)
        gs = fig_ref.add_gridspec(2, 2)
        axA = fig_ref.add_subplot(gs[0, 0])
        axA.set_title(labels[0])
        axB = fig_ref.add_subplot(gs[1, 1])
        axB.set_title(labels[1])

    def test_fail_list_of_str(self):
        with pytest.raises(ValueError, match='must be 2D'):
            plt.subplot_mosaic(['foo', 'bar'])
        with pytest.raises(ValueError, match='must be 2D'):
            plt.subplot_mosaic(['foo'])
        with pytest.raises(ValueError, match='must be 2D'):
            plt.subplot_mosaic([['foo', ('bar',)]])
        with pytest.raises(ValueError, match='must be 2D'):
            plt.subplot_mosaic([['a', 'b'], [('a', 'b'), 'c']])

    @check_figures_equal(extensions=['png'])
    @pytest.mark.parametrize('subplot_kw', [{}, {'projection': 'polar'}, None])
    def test_subplot_kw(self, fig_test, fig_ref, subplot_kw):
        x = [[1, 2]]
        grid_axes = fig_test.subplot_mosaic(x, subplot_kw=subplot_kw)
        subplot_kw = subplot_kw or {}
        gs = fig_ref.add_gridspec(1, 2)
        axA = fig_ref.add_subplot(gs[0, 0], **subplot_kw)
        axB = fig_ref.add_subplot(gs[0, 1], **subplot_kw)

    @check_figures_equal(extensions=['png'])
    @pytest.mark.parametrize('multi_value', ['BC', tuple('BC')])
    def test_per_subplot_kw(self, fig_test, fig_ref, multi_value):
        x = 'AB;CD'
        grid_axes = fig_test.subplot_mosaic(x, subplot_kw={'facecolor': 'red'}, per_subplot_kw={'D': {'facecolor': 'blue'}, multi_value: {'facecolor': 'green'}})
        gs = fig_ref.add_gridspec(2, 2)
        for color, spec in zip(['red', 'green', 'green', 'blue'], gs):
            fig_ref.add_subplot(spec, facecolor=color)

    def test_string_parser(self):
        normalize = Figure._normalize_grid_string
        assert normalize('ABC') == [['A', 'B', 'C']]
        assert normalize('AB;CC') == [['A', 'B'], ['C', 'C']]
        assert normalize('AB;CC;DE') == [['A', 'B'], ['C', 'C'], ['D', 'E']]
        assert normalize('\n                         ABC\n                         ') == [['A', 'B', 'C']]
        assert normalize('\n                         AB\n                         CC\n                         ') == [['A', 'B'], ['C', 'C']]
        assert normalize('\n                         AB\n                         CC\n                         DE\n                         ') == [['A', 'B'], ['C', 'C'], ['D', 'E']]

    def test_per_subplot_kw_expander(self):
        normalize = Figure._norm_per_subplot_kw
        assert normalize({'A': {}, 'B': {}}) == {'A': {}, 'B': {}}
        assert normalize({('A', 'B'): {}}) == {'A': {}, 'B': {}}
        with pytest.raises(ValueError, match=f'The key {'B'!r} appears multiple times'):
            normalize({('A', 'B'): {}, 'B': {}})
        with pytest.raises(ValueError, match=f'The key {'B'!r} appears multiple times'):
            normalize({'B': {}, ('A', 'B'): {}})

    def test_extra_per_subplot_kw(self):
        with pytest.raises(ValueError, match=f'The keys {set('B')!r} are in'):
            Figure().subplot_mosaic('A', per_subplot_kw={'B': {}})

    @check_figures_equal(extensions=['png'])
    @pytest.mark.parametrize('str_pattern', ['AAA\nBBB', '\nAAA\nBBB\n', 'ABC\nDEF'])
    def test_single_str_input(self, fig_test, fig_ref, str_pattern):
        grid_axes = fig_test.subplot_mosaic(str_pattern)
        grid_axes = fig_ref.subplot_mosaic([list(ln) for ln in str_pattern.strip().split('\n')])

    @pytest.mark.parametrize('x,match', [([['A', '.'], ['.', 'A']], '(?m)we found that the label .A. specifies a ' + 'non-rectangular or non-contiguous area.'), ([['A', 'B'], [None, [['A', 'B'], ['C', 'D']]]], 'There are duplicate keys .* between the outer layout'), ('AAA\nc\nBBB', 'All of the rows must be the same length'), ([['A', [['B', 'C'], ['D']]], ['E', 'E']], 'All of the rows must be the same length')])
    def test_fail(self, x, match):
        fig = plt.figure()
        with pytest.raises(ValueError, match=match):
            fig.subplot_mosaic(x)

    @check_figures_equal(extensions=['png'])
    def test_hashable_keys(self, fig_test, fig_ref):
        fig_test.subplot_mosaic([[object(), object()]])
        fig_ref.subplot_mosaic([['A', 'B']])

    @pytest.mark.parametrize('str_pattern', ['abc', 'cab', 'bca', 'cba', 'acb', 'bac'])
    def test_user_order(self, str_pattern):
        fig = plt.figure()
        ax_dict = fig.subplot_mosaic(str_pattern)
        assert list(str_pattern) == list(ax_dict)
        assert list(fig.axes) == list(ax_dict.values())

    def test_nested_user_order(self):
        layout = [['A', [['B', 'C'], ['D', 'E']]], ['F', 'G'], ['.', [['H', [['I'], ['.']]]]]]
        fig = plt.figure()
        ax_dict = fig.subplot_mosaic(layout)
        assert list(ax_dict) == list('ABCDEFGHI')
        assert list(fig.axes) == list(ax_dict.values())

    def test_share_all(self):
        layout = [['A', [['B', 'C'], ['D', 'E']]], ['F', 'G'], ['.', [['H', [['I'], ['.']]]]]]
        fig = plt.figure()
        ax_dict = fig.subplot_mosaic(layout, sharex=True, sharey=True)
        ax_dict['A'].set(xscale='log', yscale='logit')
        assert all((ax.get_xscale() == 'log' and ax.get_yscale() == 'logit' for ax in ax_dict.values()))