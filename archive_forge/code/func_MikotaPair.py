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
def MikotaPair(n):
    """Build a pair of full diagonal matrices for the generalized eigenvalue
    problem. The Mikota pair acts as a nice test since the eigenvalues are the
    squares of the integers n, n=1,2,...
    """
    x = np.arange(1, n + 1)
    B = diag(1.0 / x)
    y = np.arange(n - 1, 0, -1)
    z = np.arange(2 * n - 1, 0, -2)
    A = diag(z) - diag(y, -1) - diag(y, 1)
    return (A, B)