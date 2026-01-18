import math
import pytest
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
from pytest import raises as assert_raises
from numpy import float32, float64, complex64, complex128, arange, triu, \
from numpy.random import rand, seed
from scipy.linalg import _fblas as fblas, get_blas_funcs, toeplitz, solve
class TestSyHe:
    """Quick and simple tests for (zc)-symm, syrk, syr2k."""

    def setup_method(self):
        self.sigma_y = np.array([[0.0, -1j], [1j, 0.0]])

    def test_symm_zc(self):
        for f in _get_func('symm', 'zc'):
            res = f(a=self.sigma_y, b=self.sigma_y, alpha=1.0)
            assert_array_almost_equal(np.triu(res), np.diag([1, -1]))

    def test_hemm_zc(self):
        for f in _get_func('hemm', 'zc'):
            res = f(a=self.sigma_y, b=self.sigma_y, alpha=1.0)
            assert_array_almost_equal(np.triu(res), np.diag([1, 1]))

    def test_syrk_zr(self):
        for f in _get_func('syrk', 'zc'):
            res = f(a=self.sigma_y, alpha=1.0)
            assert_array_almost_equal(np.triu(res), np.diag([-1, -1]))

    def test_herk_zr(self):
        for f in _get_func('herk', 'zc'):
            res = f(a=self.sigma_y, alpha=1.0)
            assert_array_almost_equal(np.triu(res), np.diag([1, 1]))

    def test_syr2k_zr(self):
        for f in _get_func('syr2k', 'zc'):
            res = f(a=self.sigma_y, b=self.sigma_y, alpha=1.0)
            assert_array_almost_equal(np.triu(res), 2.0 * np.diag([-1, -1]))

    def test_her2k_zr(self):
        for f in _get_func('her2k', 'zc'):
            res = f(a=self.sigma_y, b=self.sigma_y, alpha=1.0)
            assert_array_almost_equal(np.triu(res), 2.0 * np.diag([1, 1]))