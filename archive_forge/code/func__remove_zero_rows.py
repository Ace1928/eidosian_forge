import numpy as np
from scipy.linalg import svd
from scipy.linalg.interpolative import interp_decomp
import scipy
from scipy.linalg.blas import dtrsm
def _remove_zero_rows(A, b):
    """
    Eliminates trivial equations from system of equations defined by Ax = b
   and identifies trivial infeasibilities

    Parameters
    ----------
    A : 2-D array
        An array representing the left-hand side of a system of equations
    b : 1-D array
        An array representing the right-hand side of a system of equations

    Returns
    -------
    A : 2-D array
        An array representing the left-hand side of a system of equations
    b : 1-D array
        An array representing the right-hand side of a system of equations
    status: int
        An integer indicating the status of the removal operation
        0: No infeasibility identified
        2: Trivially infeasible
    message : str
        A string descriptor of the exit status of the optimization.

    """
    status = 0
    message = ''
    i_zero = _row_count(A) == 0
    A = A[np.logical_not(i_zero), :]
    if not np.allclose(b[i_zero], 0):
        status = 2
        message = 'There is a zero row in A_eq with a nonzero corresponding entry in b_eq. The problem is infeasible.'
    b = b[np.logical_not(i_zero)]
    return (A, b, status, message)