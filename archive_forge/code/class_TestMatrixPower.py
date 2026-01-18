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
@pytest.mark.parametrize('dt', [np.dtype(c) for c in '?bBhHiIqQefdgFDGO'])
class TestMatrixPower:
    rshft_0 = np.eye(4)
    rshft_1 = rshft_0[[3, 0, 1, 2]]
    rshft_2 = rshft_0[[2, 3, 0, 1]]
    rshft_3 = rshft_0[[1, 2, 3, 0]]
    rshft_all = [rshft_0, rshft_1, rshft_2, rshft_3]
    noninv = array([[1, 0], [0, 0]])
    stacked = np.block([[[rshft_0]]] * 2)
    dtnoinv = [object, np.dtype('e'), np.dtype('g'), np.dtype('G')]

    def test_large_power(self, dt):
        rshft = self.rshft_1.astype(dt)
        assert_equal(matrix_power(rshft, 2 ** 100 + 2 ** 10 + 2 ** 5 + 0), self.rshft_0)
        assert_equal(matrix_power(rshft, 2 ** 100 + 2 ** 10 + 2 ** 5 + 1), self.rshft_1)
        assert_equal(matrix_power(rshft, 2 ** 100 + 2 ** 10 + 2 ** 5 + 2), self.rshft_2)
        assert_equal(matrix_power(rshft, 2 ** 100 + 2 ** 10 + 2 ** 5 + 3), self.rshft_3)

    def test_power_is_zero(self, dt):

        def tz(M):
            mz = matrix_power(M, 0)
            assert_equal(mz, identity_like_generalized(M))
            assert_equal(mz.dtype, M.dtype)
        for mat in self.rshft_all:
            tz(mat.astype(dt))
            if dt != object:
                tz(self.stacked.astype(dt))

    def test_power_is_one(self, dt):

        def tz(mat):
            mz = matrix_power(mat, 1)
            assert_equal(mz, mat)
            assert_equal(mz.dtype, mat.dtype)
        for mat in self.rshft_all:
            tz(mat.astype(dt))
            if dt != object:
                tz(self.stacked.astype(dt))

    def test_power_is_two(self, dt):

        def tz(mat):
            mz = matrix_power(mat, 2)
            mmul = matmul if mat.dtype != object else dot
            assert_equal(mz, mmul(mat, mat))
            assert_equal(mz.dtype, mat.dtype)
        for mat in self.rshft_all:
            tz(mat.astype(dt))
            if dt != object:
                tz(self.stacked.astype(dt))

    def test_power_is_minus_one(self, dt):

        def tz(mat):
            invmat = matrix_power(mat, -1)
            mmul = matmul if mat.dtype != object else dot
            assert_almost_equal(mmul(invmat, mat), identity_like_generalized(mat))
        for mat in self.rshft_all:
            if dt not in self.dtnoinv:
                tz(mat.astype(dt))

    def test_exceptions_bad_power(self, dt):
        mat = self.rshft_0.astype(dt)
        assert_raises(TypeError, matrix_power, mat, 1.5)
        assert_raises(TypeError, matrix_power, mat, [1])

    def test_exceptions_non_square(self, dt):
        assert_raises(LinAlgError, matrix_power, np.array([1], dt), 1)
        assert_raises(LinAlgError, matrix_power, np.array([[1], [2]], dt), 1)
        assert_raises(LinAlgError, matrix_power, np.ones((4, 3, 2), dt), 1)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_exceptions_not_invertible(self, dt):
        if dt in self.dtnoinv:
            return
        mat = self.noninv.astype(dt)
        assert_raises(LinAlgError, matrix_power, mat, -1)