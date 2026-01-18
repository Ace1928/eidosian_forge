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
class TestStrMethodFormatter:
    test_data = [('{x:05d}', (2,), '00002'), ('{x:03d}-{pos:02d}', (2, 1), '002-01')]

    @pytest.mark.parametrize('format, input, expected', test_data)
    def test_basic(self, format, input, expected):
        fmt = mticker.StrMethodFormatter(format)
        assert fmt(*input) == expected