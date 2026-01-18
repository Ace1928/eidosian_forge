import warnings
import numpy as np
from numpy.linalg import inv, LinAlgError, norm, cond, svd
from ._basic import solve, solve_triangular, matrix_balance
from .lapack import get_lapack_funcs
from ._decomp_schur import schur
from ._decomp_lu import lu
from ._decomp_qr import qr
from ._decomp_qz import ordqz
from ._decomp import _asarray_validated
from ._special_matrices import kron, block_diag
def _are_validate_args(a, b, q, r, e, s, eq_type='care'):
    """
    A helper function to validate the arguments supplied to the
    Riccati equation solvers. Any discrepancy found in the input
    matrices leads to a ``ValueError`` exception.

    Essentially, it performs:

        - a check whether the input is free of NaN and Infs
        - a pass for the data through ``numpy.atleast_2d()``
        - squareness check of the relevant arrays
        - shape consistency check of the arrays
        - singularity check of the relevant arrays
        - symmetricity check of the relevant matrices
        - a check whether the regular or the generalized version is asked.

    This function is used by ``solve_continuous_are`` and
    ``solve_discrete_are``.

    Parameters
    ----------
    a, b, q, r, e, s : array_like
        Input data
    eq_type : str
        Accepted arguments are 'care' and 'dare'.

    Returns
    -------
    a, b, q, r, e, s : ndarray
        Regularized input data
    m, n : int
        shape of the problem
    r_or_c : type
        Data type of the problem, returns float or complex
    gen_or_not : bool
        Type of the equation, True for generalized and False for regular ARE.

    """
    if eq_type.lower() not in ('dare', 'care'):
        raise ValueError("Equation type unknown. Only 'care' and 'dare' is understood")
    a = np.atleast_2d(_asarray_validated(a, check_finite=True))
    b = np.atleast_2d(_asarray_validated(b, check_finite=True))
    q = np.atleast_2d(_asarray_validated(q, check_finite=True))
    r = np.atleast_2d(_asarray_validated(r, check_finite=True))
    r_or_c = complex if np.iscomplexobj(b) else float
    for ind, mat in enumerate((a, q, r)):
        if np.iscomplexobj(mat):
            r_or_c = complex
        if not np.equal(*mat.shape):
            raise ValueError('Matrix {} should be square.'.format('aqr'[ind]))
    m, n = b.shape
    if m != a.shape[0]:
        raise ValueError('Matrix a and b should have the same number of rows.')
    if m != q.shape[0]:
        raise ValueError('Matrix a and q should have the same shape.')
    if n != r.shape[0]:
        raise ValueError('Matrix b and r should have the same number of cols.')
    for ind, mat in enumerate((q, r)):
        if norm(mat - mat.conj().T, 1) > np.spacing(norm(mat, 1)) * 100:
            raise ValueError('Matrix {} should be symmetric/hermitian.'.format('qr'[ind]))
    if eq_type == 'care':
        min_sv = svd(r, compute_uv=False)[-1]
        if min_sv == 0.0 or min_sv < np.spacing(1.0) * norm(r, 1):
            raise ValueError('Matrix r is numerically singular.')
    generalized_case = e is not None or s is not None
    if generalized_case:
        if e is not None:
            e = np.atleast_2d(_asarray_validated(e, check_finite=True))
            if not np.equal(*e.shape):
                raise ValueError('Matrix e should be square.')
            if m != e.shape[0]:
                raise ValueError('Matrix a and e should have the same shape.')
            min_sv = svd(e, compute_uv=False)[-1]
            if min_sv == 0.0 or min_sv < np.spacing(1.0) * norm(e, 1):
                raise ValueError('Matrix e is numerically singular.')
            if np.iscomplexobj(e):
                r_or_c = complex
        if s is not None:
            s = np.atleast_2d(_asarray_validated(s, check_finite=True))
            if s.shape != b.shape:
                raise ValueError('Matrix b and s should have the same shape.')
            if np.iscomplexobj(s):
                r_or_c = complex
    return (a, b, q, r, e, s, m, n, r_or_c, generalized_case)