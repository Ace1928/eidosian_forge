import os
import numpy as np
from .arpack import _arpack  # type: ignore[attr-defined]
from . import eigsh
from scipy._lib._util import check_random_state
from scipy.sparse.linalg._interface import LinearOperator, aslinearoperator
from scipy.sparse.linalg._eigen.lobpcg import lobpcg  # type: ignore[no-redef]
from scipy.linalg import svd
def _iv(A, k, ncv, tol, which, v0, maxiter, return_singular, solver, random_state):
    solver = str(solver).lower()
    solvers = {'arpack', 'lobpcg', 'propack'}
    if solver not in solvers:
        raise ValueError(f'solver must be one of {solvers}.')
    A = aslinearoperator(A)
    if not (np.issubdtype(A.dtype, np.complexfloating) or np.issubdtype(A.dtype, np.floating)):
        message = '`A` must be of floating or complex floating data type.'
        raise ValueError(message)
    if np.prod(A.shape) == 0:
        message = '`A` must not be empty.'
        raise ValueError(message)
    kmax = min(A.shape) if solver == 'propack' else min(A.shape) - 1
    if int(k) != k or not 0 < k <= kmax:
        message = '`k` must be an integer satisfying `0 < k < min(A.shape)`.'
        raise ValueError(message)
    k = int(k)
    if solver == 'arpack' and ncv is not None:
        if int(ncv) != ncv or not k < ncv < min(A.shape):
            message = '`ncv` must be an integer satisfying `k < ncv < min(A.shape)`.'
            raise ValueError(message)
        ncv = int(ncv)
    if tol < 0 or not np.isfinite(tol):
        message = '`tol` must be a non-negative floating point value.'
        raise ValueError(message)
    tol = float(tol)
    which = str(which).upper()
    whichs = {'LM', 'SM'}
    if which not in whichs:
        raise ValueError(f'`which` must be in {whichs}.')
    if v0 is not None:
        v0 = np.atleast_1d(v0)
        if not (np.issubdtype(v0.dtype, np.complexfloating) or np.issubdtype(v0.dtype, np.floating)):
            message = '`v0` must be of floating or complex floating data type.'
            raise ValueError(message)
        shape = (A.shape[0],) if solver == 'propack' else (min(A.shape),)
        if v0.shape != shape:
            message = f'`v0` must have shape {shape}.'
            raise ValueError(message)
    if maxiter is not None and (int(maxiter) != maxiter or maxiter <= 0):
        message = '`maxiter` must be a positive integer.'
        raise ValueError(message)
    maxiter = int(maxiter) if maxiter is not None else maxiter
    rs_options = {True, False, 'vh', 'u'}
    if return_singular not in rs_options:
        raise ValueError(f'`return_singular_vectors` must be in {rs_options}.')
    random_state = check_random_state(random_state)
    return (A, k, ncv, tol, which, v0, maxiter, return_singular, solver, random_state)