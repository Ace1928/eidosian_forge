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
class TestIndexLocator:

    def test_set_params(self):
        """
        Create index locator with 3 base, 4 offset. and change it to something
        else. See if change was successful.
        Should not exception.
        """
        index = mticker.IndexLocator(base=3, offset=4)
        index.set_params(base=7, offset=7)
        assert index._base == 7
        assert index.offset == 7