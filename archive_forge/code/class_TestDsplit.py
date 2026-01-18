import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
class TestDsplit:

    def test_non_iterable(self):
        assert_raises(ValueError, dsplit, 1, 1)

    def test_0D_array(self):
        a = np.array(1)
        assert_raises(ValueError, dsplit, a, 2)

    def test_1D_array(self):
        a = np.array([1, 2, 3, 4])
        assert_raises(ValueError, dsplit, a, 2)

    def test_2D_array(self):
        a = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
        try:
            dsplit(a, 2)
            assert_(0)
        except ValueError:
            pass

    def test_3D_array(self):
        a = np.array([[[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]]])
        res = dsplit(a, 2)
        desired = [np.array([[[1, 2], [1, 2]], [[1, 2], [1, 2]]]), np.array([[[3, 4], [3, 4]], [[3, 4], [3, 4]]])]
        compare_results(res, desired)