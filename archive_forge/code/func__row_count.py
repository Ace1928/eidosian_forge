import numpy as np
from scipy.linalg import svd
from scipy.linalg.interpolative import interp_decomp
import scipy
from scipy.linalg.blas import dtrsm
def _row_count(A):
    """
    Counts the number of nonzeros in each row of input array A.
    Nonzeros are defined as any element with absolute value greater than
    tol = 1e-13. This value should probably be an input to the function.

    Parameters
    ----------
    A : 2-D array
        An array representing a matrix

    Returns
    -------
    rowcount : 1-D array
        Number of nonzeros in each row of A

    """
    tol = 1e-13
    return np.array((abs(A) > tol).sum(axis=1)).flatten()