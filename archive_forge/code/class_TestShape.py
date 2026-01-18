import collections.abc
import numpy as np
from numpy import matrix, asmatrix, bmat
from numpy.testing import (
from numpy.linalg import matrix_power
from numpy.matrixlib import mat
class TestShape:
    a = np.array([[1], [2]])
    m = matrix([[1], [2]])

    def test_shape(self):
        assert_equal(self.a.shape, (2, 1))
        assert_equal(self.m.shape, (2, 1))

    def test_numpy_ravel(self):
        assert_equal(np.ravel(self.a).shape, (2,))
        assert_equal(np.ravel(self.m).shape, (2,))

    def test_member_ravel(self):
        assert_equal(self.a.ravel().shape, (2,))
        assert_equal(self.m.ravel().shape, (1, 2))

    def test_member_flatten(self):
        assert_equal(self.a.flatten().shape, (2,))
        assert_equal(self.m.flatten().shape, (1, 2))

    def test_numpy_ravel_order(self):
        x = np.array([[1, 2, 3], [4, 5, 6]])
        assert_equal(np.ravel(x), [1, 2, 3, 4, 5, 6])
        assert_equal(np.ravel(x, order='F'), [1, 4, 2, 5, 3, 6])
        assert_equal(np.ravel(x.T), [1, 4, 2, 5, 3, 6])
        assert_equal(np.ravel(x.T, order='A'), [1, 2, 3, 4, 5, 6])
        x = matrix([[1, 2, 3], [4, 5, 6]])
        assert_equal(np.ravel(x), [1, 2, 3, 4, 5, 6])
        assert_equal(np.ravel(x, order='F'), [1, 4, 2, 5, 3, 6])
        assert_equal(np.ravel(x.T), [1, 4, 2, 5, 3, 6])
        assert_equal(np.ravel(x.T, order='A'), [1, 2, 3, 4, 5, 6])

    def test_matrix_ravel_order(self):
        x = matrix([[1, 2, 3], [4, 5, 6]])
        assert_equal(x.ravel(), [[1, 2, 3, 4, 5, 6]])
        assert_equal(x.ravel(order='F'), [[1, 4, 2, 5, 3, 6]])
        assert_equal(x.T.ravel(), [[1, 4, 2, 5, 3, 6]])
        assert_equal(x.T.ravel(order='A'), [[1, 2, 3, 4, 5, 6]])

    def test_array_memory_sharing(self):
        assert_(np.may_share_memory(self.a, self.a.ravel()))
        assert_(not np.may_share_memory(self.a, self.a.flatten()))

    def test_matrix_memory_sharing(self):
        assert_(np.may_share_memory(self.m, self.m.ravel()))
        assert_(not np.may_share_memory(self.m, self.m.flatten()))

    def test_expand_dims_matrix(self):
        a = np.arange(10).reshape((2, 5)).view(np.matrix)
        expanded = np.expand_dims(a, axis=1)
        assert_equal(expanded.ndim, 3)
        assert_(not isinstance(expanded, np.matrix))