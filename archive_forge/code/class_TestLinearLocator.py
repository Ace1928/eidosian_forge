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
class TestLinearLocator:

    def test_basic(self):
        loc = mticker.LinearLocator(numticks=3)
        test_value = np.array([-0.8, -0.3, 0.2])
        assert_almost_equal(loc.tick_values(-0.8, 0.2), test_value)

    def test_zero_numticks(self):
        loc = mticker.LinearLocator(numticks=0)
        loc.tick_values(-0.8, 0.2) == []

    def test_set_params(self):
        """
        Create linear locator with presets={}, numticks=2 and change it to
        something else. See if change was successful. Should not exception.
        """
        loc = mticker.LinearLocator(numticks=2)
        loc.set_params(numticks=8, presets={(0, 1): []})
        assert loc.numticks == 8
        assert loc.presets == {(0, 1): []}

    def test_presets(self):
        loc = mticker.LinearLocator(presets={(1, 2): [1, 1.25, 1.75], (0, 2): [0.5, 1.5]})
        assert loc.tick_values(1, 2) == [1, 1.25, 1.75]
        assert loc.tick_values(2, 1) == [1, 1.25, 1.75]
        assert loc.tick_values(0, 2) == [0.5, 1.5]
        assert loc.tick_values(0.0, 2.0) == [0.5, 1.5]
        assert (loc.tick_values(0, 1) == np.linspace(0, 1, 11)).all()