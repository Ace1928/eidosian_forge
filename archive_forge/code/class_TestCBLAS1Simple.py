import math
import pytest
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
from pytest import raises as assert_raises
from numpy import float32, float64, complex64, complex128, arange, triu, \
from numpy.random import rand, seed
from scipy.linalg import _fblas as fblas, get_blas_funcs, toeplitz, solve
class TestCBLAS1Simple:

    def test_axpy(self):
        for p in 'sd':
            f = getattr(cblas, p + 'axpy', None)
            if f is None:
                continue
            assert_array_almost_equal(f([1, 2, 3], [2, -1, 3], a=5), [7, 9, 18])
        for p in 'cz':
            f = getattr(cblas, p + 'axpy', None)
            if f is None:
                continue
            assert_array_almost_equal(f([1, 2j, 3], [2, -1, 3], a=5), [7, 10j - 1, 18])