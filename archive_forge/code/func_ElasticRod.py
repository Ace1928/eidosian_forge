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
def ElasticRod(n):
    """Build the matrices for the generalized eigenvalue problem of the
    fixed-free elastic rod vibration model.
    """
    L = 1.0
    le = L / n
    rho = 7850.0
    S = 0.0001
    E = 210000000000.0
    mass = rho * S * le / 6.0
    k = E * S / le
    A = k * (diag(r_[2.0 * ones(n - 1), 1]) - diag(ones(n - 1), 1) - diag(ones(n - 1), -1))
    B = mass * (diag(r_[4.0 * ones(n - 1), 2]) + diag(ones(n - 1), 1) + diag(ones(n - 1), -1))
    return (A, B)