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

    