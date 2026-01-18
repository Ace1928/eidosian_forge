import pytest
import numpy as np
from numpy.testing import (
from numpy.lib.index_tricks import (
class TestConcatenator:

    def test_1d(self):
        assert_array_equal(r_[1, 2, 3, 4, 5, 6], np.array([1, 2, 3, 4, 5, 6]))
        b = np.ones(5)
        c = r_[b, 0, 0, b]
        assert_array_equal(c, [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1])

    def test_mixed_type(self):
        g = r_[10.1, 1:10]
        assert_(g.dtype == 'f8')

    def test_more_mixed_type(self):
        g = r_[-10.1, np.array([1]), np.array([2, 3, 4]), 10.0]
        assert_(g.dtype == 'f8')

    def test_complex_step(self):
        g = r_[0:36:100j]
        assert_(g.shape == (100,))
        g = r_[0:36:np.complex64(100j)]
        assert_(g.shape == (100,))

    def test_2d(self):
        b = np.random.rand(5, 5)
        c = np.random.rand(5, 5)
        d = r_['1', b, c]
        assert_(d.shape == (5, 10))
        assert_array_equal(d[:, :5], b)
        assert_array_equal(d[:, 5:], c)
        d = r_[b, c]
        assert_(d.shape == (10, 5))
        assert_array_equal(d[:5, :], b)
        assert_array_equal(d[5:, :], c)

    def test_0d(self):
        assert_equal(r_[0, np.array(1), 2], [0, 1, 2])
        assert_equal(r_[[0, 1, 2], np.array(3)], [0, 1, 2, 3])
        assert_equal(r_[np.array(0), [1, 2, 3]], [0, 1, 2, 3])