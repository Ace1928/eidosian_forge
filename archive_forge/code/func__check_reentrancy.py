import itertools
import platform
import sys
import numpy as np
from numpy.testing import (assert_array_equal, assert_allclose,
import pytest
from numpy import zeros, arange, array, ones, eye, iscomplexobj
from scipy.linalg import norm
from scipy.sparse import spdiags, csr_matrix, SparseEfficiencyWarning, kronsum
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg._isolve import (bicg, bicgstab, cg, cgs,
def _check_reentrancy(solver, is_reentrant):

    def matvec(x):
        A = np.array([[1.0, 0, 0], [0, 2.0, 0], [0, 0, 3.0]])
        y, info = solver(A, x)
        assert info == 0
        return y
    b = np.array([1, 1.0 / 2, 1.0 / 3])
    op = LinearOperator((3, 3), matvec=matvec, rmatvec=matvec, dtype=b.dtype)
    if not is_reentrant:
        pytest.raises(RuntimeError, solver, op, b)
    else:
        y, info = solver(op, b)
        assert info == 0
        assert_allclose(y, [1, 1, 1])