import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
class TestNotMasked:

    def test_edges(self):
        data = masked_array(np.arange(25).reshape(5, 5), mask=[[0, 0, 1, 0, 0], [0, 0, 0, 1, 1], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 0, 0]])
        test = notmasked_edges(data, None)
        assert_equal(test, [0, 24])
        test = notmasked_edges(data, 0)
        assert_equal(test[0], [(0, 0, 1, 0, 0), (0, 1, 2, 3, 4)])
        assert_equal(test[1], [(3, 3, 3, 4, 4), (0, 1, 2, 3, 4)])
        test = notmasked_edges(data, 1)
        assert_equal(test[0], [(0, 1, 2, 3, 4), (0, 0, 2, 0, 3)])
        assert_equal(test[1], [(0, 1, 2, 3, 4), (4, 2, 4, 4, 4)])
        test = notmasked_edges(data.data, None)
        assert_equal(test, [0, 24])
        test = notmasked_edges(data.data, 0)
        assert_equal(test[0], [(0, 0, 0, 0, 0), (0, 1, 2, 3, 4)])
        assert_equal(test[1], [(4, 4, 4, 4, 4), (0, 1, 2, 3, 4)])
        test = notmasked_edges(data.data, -1)
        assert_equal(test[0], [(0, 1, 2, 3, 4), (0, 0, 0, 0, 0)])
        assert_equal(test[1], [(0, 1, 2, 3, 4), (4, 4, 4, 4, 4)])
        data[-2] = masked
        test = notmasked_edges(data, 0)
        assert_equal(test[0], [(0, 0, 1, 0, 0), (0, 1, 2, 3, 4)])
        assert_equal(test[1], [(1, 1, 2, 4, 4), (0, 1, 2, 3, 4)])
        test = notmasked_edges(data, -1)
        assert_equal(test[0], [(0, 1, 2, 4), (0, 0, 2, 3)])
        assert_equal(test[1], [(0, 1, 2, 4), (4, 2, 4, 4)])

    def test_contiguous(self):
        a = masked_array(np.arange(24).reshape(3, 8), mask=[[0, 0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 1, 0]])
        tmp = notmasked_contiguous(a, None)
        assert_equal(tmp, [slice(0, 4, None), slice(16, 22, None), slice(23, 24, None)])
        tmp = notmasked_contiguous(a, 0)
        assert_equal(tmp, [[slice(0, 1, None), slice(2, 3, None)], [slice(0, 1, None), slice(2, 3, None)], [slice(0, 1, None), slice(2, 3, None)], [slice(0, 1, None), slice(2, 3, None)], [slice(2, 3, None)], [slice(2, 3, None)], [], [slice(2, 3, None)]])
        tmp = notmasked_contiguous(a, 1)
        assert_equal(tmp, [[slice(0, 4, None)], [], [slice(0, 6, None), slice(7, 8, None)]])