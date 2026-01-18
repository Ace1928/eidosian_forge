import os
import re
import copy
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_equal
import pytest
from scipy.linalg import svd, null_space
from scipy.sparse import csc_matrix, issparse, spdiags, random
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg import svds
from scipy.sparse.linalg._eigen.arpack import ArpackNoConvergence
def _check_svds(A, k, u, s, vh, which='LM', check_usvh_A=False, check_svd=True, atol=1e-10, rtol=1e-07):
    n, m = A.shape
    assert_equal(u.shape, (n, k))
    assert_equal(s.shape, (k,))
    assert_equal(vh.shape, (k, m))
    A_rebuilt = (u * s).dot(vh)
    assert_equal(A_rebuilt.shape, A.shape)
    if check_usvh_A:
        assert_allclose(A_rebuilt, A, atol=atol, rtol=rtol)
    uh_u = np.dot(u.T.conj(), u)
    assert_equal(uh_u.shape, (k, k))
    assert_allclose(uh_u, np.identity(k), atol=atol, rtol=rtol)
    vh_v = np.dot(vh, vh.T.conj())
    assert_equal(vh_v.shape, (k, k))
    assert_allclose(vh_v, np.identity(k), atol=atol, rtol=rtol)
    if check_svd:
        u2, s2, vh2 = sorted_svd(A, k, which)
        assert_allclose(np.abs(u), np.abs(u2), atol=atol, rtol=rtol)
        assert_allclose(s, s2, atol=atol, rtol=rtol)
        assert_allclose(np.abs(vh), np.abs(vh2), atol=atol, rtol=rtol)