import itertools
import warnings
import numpy as np
from numpy import (arange, array, dot, zeros, identity, conjugate, transpose,
from numpy.random import random
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
import pytest
from pytest import raises as assert_raises
from scipy.linalg import (solve, inv, det, lstsq, pinv, pinvh, norm,
from scipy.linalg._testutils import assert_no_overwrite
from scipy._lib._testutils import check_free_memory, IS_MUSL
from scipy.linalg.blas import HAS_ILP64
from scipy._lib.deprecation import _NoValue
class TestInv:

    def setup_method(self):
        np.random.seed(1234)

    def test_simple(self):
        a = [[1, 2], [3, 4]]
        a_inv = inv(a)
        assert_array_almost_equal(dot(a, a_inv), np.eye(2))
        a = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
        a_inv = inv(a)
        assert_array_almost_equal(dot(a, a_inv), np.eye(3))

    def test_random(self):
        n = 20
        for i in range(4):
            a = random([n, n])
            for i in range(n):
                a[i, i] = 20 * (0.1 + a[i, i])
            a_inv = inv(a)
            assert_array_almost_equal(dot(a, a_inv), identity(n))

    def test_simple_complex(self):
        a = [[1, 2], [3, 4j]]
        a_inv = inv(a)
        assert_array_almost_equal(dot(a, a_inv), [[1, 0], [0, 1]])

    def test_random_complex(self):
        n = 20
        for i in range(4):
            a = random([n, n]) + 2j * random([n, n])
            for i in range(n):
                a[i, i] = 20 * (0.1 + a[i, i])
            a_inv = inv(a)
            assert_array_almost_equal(dot(a, a_inv), identity(n))

    def test_check_finite(self):
        a = [[1, 2], [3, 4]]
        a_inv = inv(a, check_finite=False)
        assert_array_almost_equal(dot(a, a_inv), [[1, 0], [0, 1]])