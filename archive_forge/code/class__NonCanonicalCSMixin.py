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
class _NonCanonicalCSMixin(_NonCanonicalCompressedMixin):

    def test_getelement(self):

        def check(dtype, sorted_indices):
            D = array([[1, 0, 0], [4, 3, 0], [0, 2, 0], [0, 0, 0]], dtype=dtype)
            A = self.spcreator(D, sorted_indices=sorted_indices)
            M, N = D.shape
            for i in range(-M, M):
                for j in range(-N, N):
                    assert_equal(A[i, j], D[i, j])
            for ij in [(0, 3), (-1, 3), (4, 0), (4, 3), (4, -1), (1, 2, 3)]:
                assert_raises((IndexError, TypeError), A.__getitem__, ij)
        for dtype in supported_dtypes:
            for sorted_indices in [False, True]:
                check(np.dtype(dtype), sorted_indices)

    def test_setitem_sparse(self):
        D = np.eye(3)
        A = self.spcreator(D)
        B = self.spcreator([[1, 2, 3]])
        D[1, :] = B.toarray()
        with suppress_warnings() as sup:
            sup.filter(SparseEfficiencyWarning, 'Changing the sparsity structure of a cs[cr]_matrix is expensive')
            A[1, :] = B
        assert_array_equal(A.toarray(), D)
        D[:, 2] = B.toarray().ravel()
        with suppress_warnings() as sup:
            sup.filter(SparseEfficiencyWarning, 'Changing the sparsity structure of a cs[cr]_matrix is expensive')
            A[:, 2] = B.T
        assert_array_equal(A.toarray(), D)

    @pytest.mark.xfail(run=False, reason='inverse broken with non-canonical matrix')
    def test_inv(self):
        pass

    @pytest.mark.xfail(run=False, reason='solve broken with non-canonical matrix')
    def test_solve(self):
        pass