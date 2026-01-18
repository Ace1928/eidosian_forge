import contextlib
import functools
import operator
import platform
import itertools
import sys
from scipy._lib import _pep440
import numpy as np
from numpy import (arange, zeros, array, dot, asarray,
import random
from numpy.testing import (assert_equal, assert_array_equal,
from pytest import raises as assert_raises
import scipy.linalg
import scipy.sparse as sparse
from scipy.sparse import (csc_matrix, csr_matrix, dok_matrix,
from scipy.sparse._sputils import (supported_dtypes, isscalarlike,
from scipy.sparse.linalg import splu, expm, inv
from scipy._lib.decorator import decorator
from scipy._lib._util import ComplexWarning
import pytest
class _TestSolve:

    def test_solve(self):
        n = 20
        np.random.seed(0)
        A = zeros((n, n), dtype=complex)
        x = np.random.rand(n)
        y = np.random.rand(n - 1) + 1j * np.random.rand(n - 1)
        r = np.random.rand(n)
        for i in range(len(x)):
            A[i, i] = x[i]
        for i in range(len(y)):
            A[i, i + 1] = y[i]
            A[i + 1, i] = conjugate(y[i])
        A = self.spcreator(A)
        with suppress_warnings() as sup:
            sup.filter(SparseEfficiencyWarning, 'splu converted its input to CSC format')
            x = splu(A).solve(r)
        assert_almost_equal(A @ x, r)