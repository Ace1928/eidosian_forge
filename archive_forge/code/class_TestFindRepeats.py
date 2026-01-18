import os
import re
import warnings
from collections import namedtuple
from itertools import product
import hypothesis.extra.numpy as npst
import hypothesis
import contextlib
from numpy.testing import (assert_, assert_equal,
import pytest
from pytest import raises as assert_raises
import numpy.ma.testutils as mat
from numpy import array, arange, float32, float64, power
import numpy as np
import scipy.stats as stats
import scipy.stats.mstats as mstats
import scipy.stats._mstats_basic as mstats_basic
from scipy.stats._ksstats import kolmogn
from scipy.special._testutils import FuncData
from scipy.special import binom
from scipy import optimize
from .common_tests import check_named_results
from scipy.spatial.distance import cdist
from scipy.stats._axis_nan_policy import _broadcast_concatenate
from scipy.stats._stats_py import _permutation_distribution_t
from scipy._lib._util import AxisError
class TestFindRepeats:

    def test_basic(self):
        a = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 5]
        res, nums = stats.find_repeats(a)
        assert_array_equal(res, [1, 2, 3, 4])
        assert_array_equal(nums, [3, 3, 2, 2])

    def test_empty_result(self):
        for a in [[10, 20, 50, 30, 40], []]:
            repeated, counts = stats.find_repeats(a)
            assert_array_equal(repeated, [])
            assert_array_equal(counts, [])