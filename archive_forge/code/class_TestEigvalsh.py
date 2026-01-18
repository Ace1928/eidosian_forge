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
class TestEigvalsh:

    @pytest.mark.parametrize('dtype', [single, double, csingle, cdouble])
    def test_types(self, dtype):
        x = np.array([[1, 0.5], [0.5, 1]], dtype=dtype)
        w = np.linalg.eigvalsh(x)
        assert_equal(w.dtype, get_real_dtype(dtype))

    def test_invalid(self):
        x = np.array([[1, 0.5], [0.5, 1]], dtype=np.float32)
        assert_raises(ValueError, np.linalg.eigvalsh, x, UPLO='lrong')
        assert_raises(ValueError, np.linalg.eigvalsh, x, 'lower')
        assert_raises(ValueError, np.linalg.eigvalsh, x, 'upper')

    def test_UPLO(self):
        Klo = np.array([[0, 0], [1, 0]], dtype=np.double)
        Kup = np.array([[0, 1], [0, 0]], dtype=np.double)
        tgt = np.array([-1, 1], dtype=np.double)
        rtol = get_rtol(np.double)
        w = np.linalg.eigvalsh(Klo)
        assert_allclose(w, tgt, rtol=rtol)
        w = np.linalg.eigvalsh(Klo, UPLO='L')
        assert_allclose(w, tgt, rtol=rtol)
        w = np.linalg.eigvalsh(Klo, UPLO='l')
        assert_allclose(w, tgt, rtol=rtol)
        w = np.linalg.eigvalsh(Kup, UPLO='U')
        assert_allclose(w, tgt, rtol=rtol)
        w = np.linalg.eigvalsh(Kup, UPLO='u')
        assert_allclose(w, tgt, rtol=rtol)

    def test_0_size(self):

        class ArraySubclass(np.ndarray):
            pass
        a = np.zeros((0, 1, 1), dtype=np.int_).view(ArraySubclass)
        res = linalg.eigvalsh(a)
        assert_(res.dtype.type is np.float64)
        assert_equal((0, 1), res.shape)
        assert_(isinstance(res, np.ndarray))
        a = np.zeros((0, 0), dtype=np.complex64).view(ArraySubclass)
        res = linalg.eigvalsh(a)
        assert_(res.dtype.type is np.float32)
        assert_equal((0,), res.shape)
        assert_(isinstance(res, np.ndarray))