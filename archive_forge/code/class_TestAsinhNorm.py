import copy
import itertools
import unittest.mock
from packaging.version import parse as parse_version
from io import BytesIO
import numpy as np
from PIL import Image
import pytest
import base64
from numpy.testing import assert_array_equal, assert_array_almost_equal
from matplotlib import cbook, cm
import matplotlib
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
import matplotlib.pyplot as plt
import matplotlib.scale as mscale
from matplotlib.rcsetup import cycler
from matplotlib.testing.decorators import image_comparison, check_figures_equal
class TestAsinhNorm:
    """
    Tests for `~.colors.AsinhNorm`
    """

    def test_init(self):
        norm0 = mcolors.AsinhNorm()
        assert norm0.linear_width == 1
        norm5 = mcolors.AsinhNorm(linear_width=5)
        assert norm5.linear_width == 5

    def test_norm(self):
        norm = mcolors.AsinhNorm(2, vmin=-4, vmax=4)
        vals = np.arange(-3.5, 3.5, 10)
        normed_vals = norm(vals)
        asinh2 = np.arcsinh(2)
        expected = (2 * np.arcsinh(vals / 2) + 2 * asinh2) / (4 * asinh2)
        assert_array_almost_equal(normed_vals, expected)