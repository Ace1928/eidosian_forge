import collections
import platform
from unittest import mock
import warnings
import numpy as np
from numpy.testing import assert_allclose
import pytest
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing._markers import needs_usetex
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import matplotlib.collections as mcollections
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerTuple
import matplotlib.legend as mlegend
from matplotlib import _api, rc_context
from matplotlib.font_manager import FontProperties
class TestLegendFunction:

    def test_legend_no_args(self):
        lines = plt.plot(range(10), label='hello world')
        with mock.patch('matplotlib.legend.Legend') as Legend:
            plt.legend()
        Legend.assert_called_with(plt.gca(), lines, ['hello world'])

    def test_legend_positional_handles_labels(self):
        lines = plt.plot(range(10))
        with mock.patch('matplotlib.legend.Legend') as Legend:
            plt.legend(lines, ['hello world'])
        Legend.assert_called_with(plt.gca(), lines, ['hello world'])

    def test_legend_positional_handles_only(self):
        lines = plt.plot(range(10))
        with pytest.raises(TypeError, match='but found an Artist'):
            plt.legend(lines)

    def test_legend_positional_labels_only(self):
        lines = plt.plot(range(10), label='hello world')
        with mock.patch('matplotlib.legend.Legend') as Legend:
            plt.legend(['foobar'])
        Legend.assert_called_with(plt.gca(), lines, ['foobar'])

    def test_legend_three_args(self):
        lines = plt.plot(range(10), label='hello world')
        with mock.patch('matplotlib.legend.Legend') as Legend:
            plt.legend(lines, ['foobar'], loc='right')
        Legend.assert_called_with(plt.gca(), lines, ['foobar'], loc='right')

    def test_legend_handler_map(self):
        lines = plt.plot(range(10), label='hello world')
        with mock.patch('matplotlib.legend._get_legend_handles_labels') as handles_labels:
            handles_labels.return_value = (lines, ['hello world'])
            plt.legend(handler_map={'1': 2})
        handles_labels.assert_called_with([plt.gca()], {'1': 2})

    def test_legend_kwargs_handles_only(self):
        fig, ax = plt.subplots()
        x = np.linspace(0, 1, 11)
        ln1, = ax.plot(x, x, label='x')
        ln2, = ax.plot(x, 2 * x, label='2x')
        ln3, = ax.plot(x, 3 * x, label='3x')
        with mock.patch('matplotlib.legend.Legend') as Legend:
            ax.legend(handles=[ln3, ln2])
        Legend.assert_called_with(ax, [ln3, ln2], ['3x', '2x'])

    def test_legend_kwargs_labels_only(self):
        fig, ax = plt.subplots()
        x = np.linspace(0, 1, 11)
        ln1, = ax.plot(x, x)
        ln2, = ax.plot(x, 2 * x)
        with mock.patch('matplotlib.legend.Legend') as Legend:
            ax.legend(labels=['x', '2x'])
        Legend.assert_called_with(ax, [ln1, ln2], ['x', '2x'])

    def test_legend_kwargs_handles_labels(self):
        fig, ax = plt.subplots()
        th = np.linspace(0, 2 * np.pi, 1024)
        lns, = ax.plot(th, np.sin(th), label='sin')
        lnc, = ax.plot(th, np.cos(th), label='cos')
        with mock.patch('matplotlib.legend.Legend') as Legend:
            ax.legend(labels=('a', 'b'), handles=(lnc, lns))
        Legend.assert_called_with(ax, (lnc, lns), ('a', 'b'))

    def test_warn_mixed_args_and_kwargs(self):
        fig, ax = plt.subplots()
        th = np.linspace(0, 2 * np.pi, 1024)
        lns, = ax.plot(th, np.sin(th), label='sin')
        lnc, = ax.plot(th, np.cos(th), label='cos')
        with pytest.warns(UserWarning) as record:
            ax.legend((lnc, lns), labels=('a', 'b'))
        assert len(record) == 1
        assert str(record[0].message) == 'You have mixed positional and keyword arguments, some input may be discarded.'

    def test_parasite(self):
        from mpl_toolkits.axes_grid1 import host_subplot
        host = host_subplot(111)
        par = host.twinx()
        p1, = host.plot([0, 1, 2], [0, 1, 2], label='Density')
        p2, = par.plot([0, 1, 2], [0, 3, 2], label='Temperature')
        with mock.patch('matplotlib.legend.Legend') as Legend:
            plt.legend()
        Legend.assert_called_with(host, [p1, p2], ['Density', 'Temperature'])