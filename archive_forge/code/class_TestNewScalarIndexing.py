import collections.abc
import numpy as np
from numpy import matrix, asmatrix, bmat
from numpy.testing import (
from numpy.linalg import matrix_power
from numpy.matrixlib import mat
class TestNewScalarIndexing:
    a = matrix([[1, 2], [3, 4]])

    def test_dimesions(self):
        a = self.a
        x = a[0]
        assert_equal(x.ndim, 2)

    def test_array_from_matrix_list(self):
        a = self.a
        x = np.array([a, a])
        assert_equal(x.shape, [2, 2, 2])

    def test_array_to_list(self):
        a = self.a
        assert_equal(a.tolist(), [[1, 2], [3, 4]])

    def test_fancy_indexing(self):
        a = self.a
        x = a[1, [0, 1, 0]]
        assert_(isinstance(x, matrix))
        assert_equal(x, matrix([[3, 4, 3]]))
        x = a[[1, 0]]
        assert_(isinstance(x, matrix))
        assert_equal(x, matrix([[3, 4], [1, 2]]))
        x = a[[[1], [0]], [[1, 0], [0, 1]]]
        assert_(isinstance(x, matrix))
        assert_equal(x, matrix([[4, 3], [1, 2]]))

    def test_matrix_element(self):
        x = matrix([[1, 2, 3], [4, 5, 6]])
        assert_equal(x[0][0], matrix([[1, 2, 3]]))
        assert_equal(x[0][0].shape, (1, 3))
        assert_equal(x[0].shape, (1, 3))
        assert_equal(x[:, 0].shape, (2, 1))
        x = matrix(0)
        assert_equal(x[0, 0], 0)
        assert_equal(x[0], 0)
        assert_equal(x[:, 0].shape, x.shape)

    def test_scalar_indexing(self):
        x = asmatrix(np.zeros((3, 2), float))
        assert_equal(x[0, 0], x[0][0])

    def test_row_column_indexing(self):
        x = asmatrix(np.eye(2))
        assert_array_equal(x[0, :], [[1, 0]])
        assert_array_equal(x[1, :], [[0, 1]])
        assert_array_equal(x[:, 0], [[1], [0]])
        assert_array_equal(x[:, 1], [[0], [1]])

    def test_boolean_indexing(self):
        A = np.arange(6)
        A.shape = (3, 2)
        x = asmatrix(A)
        assert_array_equal(x[:, np.array([True, False])], x[:, 0])
        assert_array_equal(x[np.array([True, False, False]), :], x[0, :])

    def test_list_indexing(self):
        A = np.arange(6)
        A.shape = (3, 2)
        x = asmatrix(A)
        assert_array_equal(x[:, [1, 0]], x[:, ::-1])
        assert_array_equal(x[[2, 1, 0], :], x[::-1, :])