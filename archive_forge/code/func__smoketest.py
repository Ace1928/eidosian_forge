import sys
import threading
import numpy as np
from numpy import array, finfo, arange, eye, all, unique, ones, dot
import numpy.random as random
from numpy.testing import (
import pytest
from pytest import raises as assert_raises
import scipy.linalg
from scipy.linalg import norm, inv
from scipy.sparse import (spdiags, SparseEfficiencyWarning, csc_matrix,
from scipy.sparse.linalg import SuperLU
from scipy.sparse.linalg._dsolve import (spsolve, use_solver, splu, spilu,
import scipy.sparse
from scipy._lib._testutils import check_free_memory
from scipy._lib._util import ComplexWarning
def _smoketest(self, spxlu, check, dtype, idx_dtype):
    if np.issubdtype(dtype, np.complexfloating):
        A = self.A + 1j * self.A.T
    else:
        A = self.A
    A = A.astype(dtype)
    A.indices = A.indices.astype(idx_dtype, copy=False)
    A.indptr = A.indptr.astype(idx_dtype, copy=False)
    lu = spxlu(A)
    rng = random.RandomState(1234)
    for k in [None, 1, 2, self.n, self.n + 2]:
        msg = f'k={k!r}'
        if k is None:
            b = rng.rand(self.n)
        else:
            b = rng.rand(self.n, k)
        if np.issubdtype(dtype, np.complexfloating):
            b = b + 1j * rng.rand(*b.shape)
        b = b.astype(dtype)
        x = lu.solve(b)
        check(A, b, x, msg)
        x = lu.solve(b, 'T')
        check(A.T, b, x, msg)
        x = lu.solve(b, 'H')
        check(A.T.conj(), b, x, msg)