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
class TestMaxNLocator:
    basic_data = [(20, 100, np.array([20.0, 40.0, 60.0, 80.0, 100.0])), (0.001, 0.0001, np.array([0.0, 0.0002, 0.0004, 0.0006, 0.0008, 0.001])), (-1000000000000000.0, 1000000000000000.0, np.array([-1000000000000000.0, -500000000000000.0, 0.0, 500000000000000.0, 1000000000000000.0])), (0, 8.5e-51, np.arange(6) * 2e-51), (-8.5e-51, 0, np.arange(-5, 1) * 2e-51)]
    integer_data = [(-0.1, 1.1, None, np.array([-1, 0, 1, 2])), (-0.1, 0.95, None, np.array([-0.25, 0, 0.25, 0.5, 0.75, 1.0])), (1, 55, [1, 1.5, 5, 6, 10], np.array([0, 15, 30, 45, 60]))]

    @pytest.mark.parametrize('vmin, vmax, expected', basic_data)
    def test_basic(self, vmin, vmax, expected):
        loc = mticker.MaxNLocator(nbins=5)
        assert_almost_equal(loc.tick_values(vmin, vmax), expected)

    @pytest.mark.parametrize('vmin, vmax, steps, expected', integer_data)
    def test_integer(self, vmin, vmax, steps, expected):
        loc = mticker.MaxNLocator(nbins=5, integer=True, steps=steps)
        assert_almost_equal(loc.tick_values(vmin, vmax), expected)

    @pytest.mark.parametrize('kwargs, errortype, match', [({'foo': 0}, TypeError, re.escape("set_params() got an unexpected keyword argument 'foo'")), ({'steps': [2, 1]}, ValueError, 'steps argument must be an increasing'), ({'steps': 2}, ValueError, 'steps argument must be an increasing'), ({'steps': [2, 11]}, ValueError, 'steps argument must be an increasing')])
    def test_errors(self, kwargs, errortype, match):
        with pytest.raises(errortype, match=match):
            mticker.MaxNLocator(**kwargs)

    @pytest.mark.parametrize('steps, result', [([1, 2, 10], [1, 2, 10]), ([2, 10], [1, 2, 10]), ([1, 2], [1, 2, 10]), ([2], [1, 2, 10])])
    def test_padding(self, steps, result):
        loc = mticker.MaxNLocator(steps=steps)
        assert (loc._steps == result).all()