import os
import sys
import itertools
import traceback
import textwrap
import subprocess
import pytest
import numpy as np
from numpy import array, single, double, csingle, cdouble, dot, identity, matmul
from numpy.core import swapaxes
from numpy import multiply, atleast_2d, inf, asarray
from numpy import linalg
from numpy.linalg import matrix_power, norm, matrix_rank, multi_dot, LinAlgError
from numpy.linalg.linalg import _multi_dot_matrix_chain_order
from numpy.testing import (
class _TestNormGeneral(_TestNormBase):

    def test_empty(self):
        assert_equal(norm([]), 0.0)
        assert_equal(norm(array([], dtype=self.dt)), 0.0)
        assert_equal(norm(atleast_2d(array([], dtype=self.dt))), 0.0)

    def test_vector_return_type(self):
        a = np.array([1, 0, 1])
        exact_types = np.typecodes['AllInteger']
        inexact_types = np.typecodes['AllFloat']
        all_types = exact_types + inexact_types
        for each_type in all_types:
            at = a.astype(each_type)
            an = norm(at, -np.inf)
            self.check_dtype(at, an)
            assert_almost_equal(an, 0.0)
            with suppress_warnings() as sup:
                sup.filter(RuntimeWarning, 'divide by zero encountered')
                an = norm(at, -1)
                self.check_dtype(at, an)
                assert_almost_equal(an, 0.0)
            an = norm(at, 0)
            self.check_dtype(at, an)
            assert_almost_equal(an, 2)
            an = norm(at, 1)
            self.check_dtype(at, an)
            assert_almost_equal(an, 2.0)
            an = norm(at, 2)
            self.check_dtype(at, an)
            assert_almost_equal(an, an.dtype.type(2.0) ** an.dtype.type(1.0 / 2.0))
            an = norm(at, 4)
            self.check_dtype(at, an)
            assert_almost_equal(an, an.dtype.type(2.0) ** an.dtype.type(1.0 / 4.0))
            an = norm(at, np.inf)
            self.check_dtype(at, an)
            assert_almost_equal(an, 1.0)

    def test_vector(self):
        a = [1, 2, 3, 4]
        b = [-1, -2, -3, -4]
        c = [-1, 2, -3, 4]

        def _test(v):
            np.testing.assert_almost_equal(norm(v), 30 ** 0.5, decimal=self.dec)
            np.testing.assert_almost_equal(norm(v, inf), 4.0, decimal=self.dec)
            np.testing.assert_almost_equal(norm(v, -inf), 1.0, decimal=self.dec)
            np.testing.assert_almost_equal(norm(v, 1), 10.0, decimal=self.dec)
            np.testing.assert_almost_equal(norm(v, -1), 12.0 / 25, decimal=self.dec)
            np.testing.assert_almost_equal(norm(v, 2), 30 ** 0.5, decimal=self.dec)
            np.testing.assert_almost_equal(norm(v, -2), (205.0 / 144) ** (-0.5), decimal=self.dec)
            np.testing.assert_almost_equal(norm(v, 0), 4, decimal=self.dec)
        for v in (a, b, c):
            _test(v)
        for v in (array(a, dtype=self.dt), array(b, dtype=self.dt), array(c, dtype=self.dt)):
            _test(v)

    def test_axis(self):
        A = array([[1, 2, 3], [4, 5, 6]], dtype=self.dt)
        for order in [None, -1, 0, 1, 2, 3, np.Inf, -np.Inf]:
            expected0 = [norm(A[:, k], ord=order) for k in range(A.shape[1])]
            assert_almost_equal(norm(A, ord=order, axis=0), expected0)
            expected1 = [norm(A[k, :], ord=order) for k in range(A.shape[0])]
            assert_almost_equal(norm(A, ord=order, axis=1), expected1)
        B = np.arange(1, 25, dtype=self.dt).reshape(2, 3, 4)
        nd = B.ndim
        for order in [None, -2, 2, -1, 1, np.Inf, -np.Inf, 'fro']:
            for axis in itertools.combinations(range(-nd, nd), 2):
                row_axis, col_axis = axis
                if row_axis < 0:
                    row_axis += nd
                if col_axis < 0:
                    col_axis += nd
                if row_axis == col_axis:
                    assert_raises(ValueError, norm, B, ord=order, axis=axis)
                else:
                    n = norm(B, ord=order, axis=axis)
                    k_index = nd - (row_axis + col_axis)
                    if row_axis < col_axis:
                        expected = [norm(B[:].take(k, axis=k_index), ord=order) for k in range(B.shape[k_index])]
                    else:
                        expected = [norm(B[:].take(k, axis=k_index).T, ord=order) for k in range(B.shape[k_index])]
                    assert_almost_equal(n, expected)

    def test_keepdims(self):
        A = np.arange(1, 25, dtype=self.dt).reshape(2, 3, 4)
        allclose_err = 'order {0}, axis = {1}'
        shape_err = 'Shape mismatch found {0}, expected {1}, order={2}, axis={3}'
        expected = norm(A, ord=None, axis=None)
        found = norm(A, ord=None, axis=None, keepdims=True)
        assert_allclose(np.squeeze(found), expected, err_msg=allclose_err.format(None, None))
        expected_shape = (1, 1, 1)
        assert_(found.shape == expected_shape, shape_err.format(found.shape, expected_shape, None, None))
        for order in [None, -1, 0, 1, 2, 3, np.Inf, -np.Inf]:
            for k in range(A.ndim):
                expected = norm(A, ord=order, axis=k)
                found = norm(A, ord=order, axis=k, keepdims=True)
                assert_allclose(np.squeeze(found), expected, err_msg=allclose_err.format(order, k))
                expected_shape = list(A.shape)
                expected_shape[k] = 1
                expected_shape = tuple(expected_shape)
                assert_(found.shape == expected_shape, shape_err.format(found.shape, expected_shape, order, k))
        for order in [None, -2, 2, -1, 1, np.Inf, -np.Inf, 'fro', 'nuc']:
            for k in itertools.permutations(range(A.ndim), 2):
                expected = norm(A, ord=order, axis=k)
                found = norm(A, ord=order, axis=k, keepdims=True)
                assert_allclose(np.squeeze(found), expected, err_msg=allclose_err.format(order, k))
                expected_shape = list(A.shape)
                expected_shape[k[0]] = 1
                expected_shape[k[1]] = 1
                expected_shape = tuple(expected_shape)
                assert_(found.shape == expected_shape, shape_err.format(found.shape, expected_shape, order, k))