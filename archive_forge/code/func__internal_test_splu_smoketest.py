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
def _internal_test_splu_smoketest(self):

    def check(A, b, x, msg=''):
        eps = np.finfo(A.dtype).eps
        r = A @ x
        assert_(abs(r - b).max() < 1000.0 * eps, msg)
    for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
        for idx_dtype in [np.int32, np.int64]:
            self._smoketest(splu, check, dtype, idx_dtype)