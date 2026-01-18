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
def _internal_test_spilu_smoketest(self):
    errors = []

    def check(A, b, x, msg=''):
        r = A @ x
        err = abs(r - b).max()
        assert_(err < 0.01, msg)
        if b.dtype in (np.float64, np.complex128):
            errors.append(err)
    for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
        for idx_dtype in [np.int32, np.int64]:
            self._smoketest(spilu, check, dtype, idx_dtype)
    assert_(max(errors) > 1e-05)