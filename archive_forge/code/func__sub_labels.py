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
def _sub_labels(self, axis, subs=()):
    """Test whether locator marks subs to be labeled."""
    fmt = axis.get_minor_formatter()
    minor_tlocs = axis.get_minorticklocs()
    fmt.set_locs(minor_tlocs)
    coefs = minor_tlocs / 10 ** np.floor(np.log10(minor_tlocs))
    label_expected = [round(c) in subs for c in coefs]
    label_test = [fmt(x) != '' for x in minor_tlocs]
    assert label_test == label_expected