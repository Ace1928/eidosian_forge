from contextlib import nullcontext
import itertools
import locale
import logging
import re
from packaging.version import parse as parse_version
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
class TestLogLocator:

    def test_basic(self):
        loc = mticker.LogLocator(numticks=5)
        with pytest.raises(ValueError):
            loc.tick_values(0, 1000)
        test_value = np.array([1e-05, 0.001, 0.1, 10.0, 1000.0, 100000.0, 10000000.0, 1000000000.0])
        assert_almost_equal(loc.tick_values(0.001, 110000.0), test_value)
        loc = mticker.LogLocator(base=2)
        test_value = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0])
        assert_almost_equal(loc.tick_values(1, 100), test_value)

    def test_polar_axes(self):
        """
        Polar axes have a different ticking logic.
        """
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.set_yscale('log')
        ax.set_ylim(1, 100)
        assert_array_equal(ax.get_yticks(), [10, 100, 1000])

    def test_switch_to_autolocator(self):
        loc = mticker.LogLocator(subs='all')
        assert_array_equal(loc.tick_values(0.45, 0.55), [0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56])
        loc = mticker.LogLocator(subs=np.arange(2, 10))
        assert 1.0 not in loc.tick_values(0.9, 20.0)
        assert 10.0 not in loc.tick_values(0.9, 20.0)

    def test_set_params(self):
        """
        Create log locator with default value, base=10.0, subs=[1.0],
        numdecs=4, numticks=15 and change it to something else.
        See if change was successful. Should not raise exception.
        """
        loc = mticker.LogLocator()
        with pytest.warns(mpl.MatplotlibDeprecationWarning, match='numdecs'):
            loc.set_params(numticks=7, numdecs=8, subs=[2.0], base=4)
        assert loc.numticks == 7
        with pytest.warns(mpl.MatplotlibDeprecationWarning, match='numdecs'):
            assert loc.numdecs == 8
        assert loc._base == 4
        assert list(loc._subs) == [2.0]

    def test_tick_values_correct(self):
        ll = mticker.LogLocator(subs=(1, 2, 5))
        test_value = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0, 50000.0, 100000.0, 200000.0, 500000.0, 1000000.0, 2000000.0, 5000000.0, 10000000.0, 20000000.0, 50000000.0, 100000000.0, 200000000.0, 500000000.0])
        assert_almost_equal(ll.tick_values(1, 10000000.0), test_value)

    def test_tick_values_not_empty(self):
        mpl.rcParams['_internal.classic_mode'] = False
        ll = mticker.LogLocator(subs=(1, 2, 5))
        test_value = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0, 50000.0, 100000.0, 200000.0, 500000.0, 1000000.0, 2000000.0, 5000000.0, 10000000.0, 20000000.0, 50000000.0, 100000000.0, 200000000.0, 500000000.0, 1000000000.0, 2000000000.0, 5000000000.0])
        assert_almost_equal(ll.tick_values(1, 100000000.0), test_value)

    def test_multiple_shared_axes(self):
        rng = np.random.default_rng(19680801)
        dummy_data = [rng.normal(size=100), [], []]
        fig, axes = plt.subplots(len(dummy_data), sharex=True, sharey=True)
        for ax, data in zip(axes.flatten(), dummy_data):
            ax.hist(data, bins=10)
            ax.set_yscale('log', nonpositive='clip')
        for ax in axes.flatten():
            assert all(ax.get_yticks() == axes[0].get_yticks())
            assert ax.get_ylim() == axes[0].get_ylim()