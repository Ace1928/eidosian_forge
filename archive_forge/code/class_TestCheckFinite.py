import operator
import warnings
import sys
import decimal
from fractions import Fraction
import math
import pytest
import hypothesis
from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st
from functools import partial
import numpy as np
from numpy import ma
from numpy.testing import (
import numpy.lib.function_base as nfb
from numpy.random import rand
from numpy.lib import (
from numpy.core.numeric import normalize_axis_tuple
class TestCheckFinite:

    def test_simple(self):
        a = [1, 2, 3]
        b = [1, 2, np.inf]
        c = [1, 2, np.nan]
        np.lib.asarray_chkfinite(a)
        assert_raises(ValueError, np.lib.asarray_chkfinite, b)
        assert_raises(ValueError, np.lib.asarray_chkfinite, c)

    def test_dtype_order(self):
        a = [1, 2, 3]
        a = np.lib.asarray_chkfinite(a, order='F', dtype=np.float64)
        assert_(a.dtype == np.float64)