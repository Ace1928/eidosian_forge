import itertools
import platform
import sys
import pytest
import numpy as np
from numpy import ones, r_, diag
from numpy.testing import (assert_almost_equal, assert_equal,
from scipy import sparse
from scipy.linalg import eig, eigh, toeplitz, orth
from scipy.sparse import spdiags, diags, eye, csr_matrix
from scipy.sparse.linalg import eigs, LinearOperator
from scipy.sparse.linalg._eigen.lobpcg import lobpcg
from scipy.sparse.linalg._eigen.lobpcg.lobpcg import _b_orthonormalize
from scipy._lib._util import np_long, np_ulong
def _check_fiedler(n, p):
    """Check the Fiedler vector computation.
    """
    col = np.zeros(n)
    col[1] = 1
    A = toeplitz(col)
    D = np.diag(A.sum(axis=1))
    L = D - A
    tmp = np.pi * np.arange(n) / n
    analytic_w = 2 * (1 - np.cos(tmp))
    analytic_V = np.cos(np.outer(np.arange(n) + 1 / 2, tmp))
    _check_eigen(L, analytic_w, analytic_V)
    eigh_w, eigh_V = eigh(L)
    _check_eigen(L, eigh_w, eigh_V)
    assert_array_less(np.abs([eigh_w[0], analytic_w[0]]), 1e-14)
    assert_allclose(eigh_w[1:], analytic_w[1:])
    X = analytic_V[:, :p]
    lobpcg_w, lobpcg_V = lobpcg(L, X, largest=False)
    assert_equal(lobpcg_w.shape, (p,))
    assert_equal(lobpcg_V.shape, (n, p))
    _check_eigen(L, lobpcg_w, lobpcg_V)
    assert_array_less(np.abs(np.min(lobpcg_w)), 1e-14)
    assert_allclose(np.sort(lobpcg_w)[1:], analytic_w[1:p])
    X = analytic_V[:, -p:]
    lobpcg_w, lobpcg_V = lobpcg(L, X, largest=True)
    assert_equal(lobpcg_w.shape, (p,))
    assert_equal(lobpcg_V.shape, (n, p))
    _check_eigen(L, lobpcg_w, lobpcg_V)
    assert_allclose(np.sort(lobpcg_w), analytic_w[-p:])
    fiedler_guess = np.concatenate((np.ones(n // 2), -np.ones(n - n // 2)))
    X = np.vstack((np.ones(n), fiedler_guess)).T
    lobpcg_w, _ = lobpcg(L, X, largest=False)
    lobpcg_w = np.sort(lobpcg_w)
    assert_allclose(lobpcg_w, analytic_w[:2], atol=1e-14)