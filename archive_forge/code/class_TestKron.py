import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
class TestKron:

    def test_basic(self):
        a = np.array(1)
        b = np.array([[1, 2], [3, 4]])
        k = np.array([[1, 2], [3, 4]])
        assert_array_equal(np.kron(a, b), k)
        a = np.array([[1, 2], [3, 4]])
        b = np.array(1)
        assert_array_equal(np.kron(a, b), k)
        a = np.array([3])
        b = np.array([[1, 2], [3, 4]])
        k = np.array([[3, 6], [9, 12]])
        assert_array_equal(np.kron(a, b), k)
        a = np.array([[1, 2], [3, 4]])
        b = np.array([3])
        assert_array_equal(np.kron(a, b), k)
        a = np.array([[[1]], [[2]]])
        b = np.array([[1, 2], [3, 4]])
        k = np.array([[[1, 2], [3, 4]], [[2, 4], [6, 8]]])
        assert_array_equal(np.kron(a, b), k)
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[[1]], [[2]]])
        k = np.array([[[1, 2], [3, 4]], [[2, 4], [6, 8]]])
        assert_array_equal(np.kron(a, b), k)

    def test_return_type(self):

        class myarray(np.ndarray):
            __array_priority__ = 1.0
        a = np.ones([2, 2])
        ma = myarray(a.shape, a.dtype, a.data)
        assert_equal(type(kron(a, a)), np.ndarray)
        assert_equal(type(kron(ma, ma)), myarray)
        assert_equal(type(kron(a, ma)), myarray)
        assert_equal(type(kron(ma, a)), myarray)

    @pytest.mark.parametrize('array_class', [np.asarray, np.mat])
    def test_kron_smoke(self, array_class):
        a = array_class(np.ones([3, 3]))
        b = array_class(np.ones([3, 3]))
        k = array_class(np.ones([9, 9]))
        assert_array_equal(np.kron(a, b), k)

    def test_kron_ma(self):
        x = np.ma.array([[1, 2], [3, 4]], mask=[[0, 1], [1, 0]])
        k = np.ma.array(np.diag([1, 4, 4, 16]), mask=~np.array(np.identity(4), dtype=bool))
        assert_array_equal(k, np.kron(x, x))

    @pytest.mark.parametrize('shape_a,shape_b', [((1, 1), (1, 1)), ((1, 2, 3), (4, 5, 6)), ((2, 2), (2, 2, 2)), ((1, 0), (1, 1)), ((2, 0, 2), (2, 2)), ((2, 0, 0, 2), (2, 0, 2))])
    def test_kron_shape(self, shape_a, shape_b):
        a = np.ones(shape_a)
        b = np.ones(shape_b)
        normalised_shape_a = (1,) * max(0, len(shape_b) - len(shape_a)) + shape_a
        normalised_shape_b = (1,) * max(0, len(shape_a) - len(shape_b)) + shape_b
        expected_shape = np.multiply(normalised_shape_a, normalised_shape_b)
        k = np.kron(a, b)
        assert np.array_equal(k.shape, expected_shape), 'Unexpected shape from kron'