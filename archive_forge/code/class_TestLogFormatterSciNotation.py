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
class TestLogFormatterSciNotation:
    test_data = [(2, 0.03125, '$\\mathdefault{2^{-5}}$'), (2, 1, '$\\mathdefault{2^{0}}$'), (2, 32, '$\\mathdefault{2^{5}}$'), (2, 0.0375, '$\\mathdefault{1.2\\times2^{-5}}$'), (2, 1.2, '$\\mathdefault{1.2\\times2^{0}}$'), (2, 38.4, '$\\mathdefault{1.2\\times2^{5}}$'), (10, -1, '$\\mathdefault{-10^{0}}$'), (10, 1e-05, '$\\mathdefault{10^{-5}}$'), (10, 1, '$\\mathdefault{10^{0}}$'), (10, 100000, '$\\mathdefault{10^{5}}$'), (10, 2e-05, '$\\mathdefault{2\\times10^{-5}}$'), (10, 2, '$\\mathdefault{2\\times10^{0}}$'), (10, 200000, '$\\mathdefault{2\\times10^{5}}$'), (10, 5e-05, '$\\mathdefault{5\\times10^{-5}}$'), (10, 5, '$\\mathdefault{5\\times10^{0}}$'), (10, 500000, '$\\mathdefault{5\\times10^{5}}$')]

    @mpl.style.context('default')
    @pytest.mark.parametrize('base, value, expected', test_data)
    def test_basic(self, base, value, expected):
        formatter = mticker.LogFormatterSciNotation(base=base)
        with mpl.rc_context({'text.usetex': False}):
            assert formatter(value) == expected