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
class Test_I0:

    def test_simple(self):
        assert_almost_equal(i0(0.5), np.array(1.0634833707413234))
        A = np.array([0.49842636, 0.6969809, 0.22011976, 0.0155549, 10.0])
        expected = np.array([1.06307822, 1.12518299, 1.01214991, 1.00006049, 2815.71662847])
        assert_almost_equal(i0(A), expected)
        assert_almost_equal(i0(-A), expected)
        B = np.array([[0.827002, 0.99959078], [0.89694769, 0.39298162], [0.37954418, 0.05206293], [0.36465447, 0.72446427], [0.48164949, 0.50324519]])
        assert_almost_equal(i0(B), np.array([[1.17843223, 1.26583466], [1.21147086, 1.0389829], [1.03633899, 1.00067775], [1.03352052, 1.13557954], [1.0588429, 1.06432317]]))
        i0_0 = np.i0([0.0])
        assert_equal(i0_0.shape, (1,))
        assert_array_equal(np.i0([0.0]), np.array([1.0]))

    def test_non_array(self):
        a = np.arange(4)

        class array_like:
            __array_interface__ = a.__array_interface__

            def __array_wrap__(self, arr):
                return self
        assert isinstance(np.abs(array_like()), array_like)
        exp = np.i0(a)
        res = np.i0(array_like())
        assert_array_equal(exp, res)

    def test_complex(self):
        a = np.array([0, 1 + 2j])
        with pytest.raises(TypeError, match='i0 not supported for complex values'):
            res = i0(a)