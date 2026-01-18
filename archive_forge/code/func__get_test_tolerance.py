import threading
import itertools
import numpy as np
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from pytest import raises as assert_raises
import pytest
from numpy import dot, conj, random
from scipy.linalg import eig, eigh
from scipy.sparse import csc_matrix, csr_matrix, diags, rand
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg._eigen.arpack import (eigs, eigsh, arpack,
from scipy._lib._gcutils import assert_deallocated, IS_PYPY
def _get_test_tolerance(type_char, mattype=None, D_type=None, which=None):
    """
    Return tolerance values suitable for a given test:

    Parameters
    ----------
    type_char : {'f', 'd', 'F', 'D'}
        Data type in ARPACK eigenvalue problem
    mattype : {csr_matrix, aslinearoperator, asarray}, optional
        Linear operator type

    Returns
    -------
    tol
        Tolerance to pass to the ARPACK routine
    rtol
        Relative tolerance for outputs
    atol
        Absolute tolerance for outputs

    """
    rtol = {'f': 3000 * np.finfo(np.float32).eps, 'F': 3000 * np.finfo(np.float32).eps, 'd': 2000 * np.finfo(np.float64).eps, 'D': 2000 * np.finfo(np.float64).eps}[type_char]
    atol = rtol
    tol = 0
    if mattype is aslinearoperator and type_char in ('f', 'F'):
        tol = 30 * np.finfo(np.float32).eps
        rtol *= 5
    if mattype is csr_matrix and type_char in ('f', 'F'):
        rtol *= 5
    if which in ('LM', 'SM', 'LA') and D_type.name == 'gen-hermitian-Mc':
        if type_char == 'F':
            rtol *= 5
        if type_char == 'D':
            rtol *= 10
            atol *= 10
    return (tol, rtol, atol)