import cvxpy.utilities.cpp.sparsecholesky as spchol  # noqa: I001
import cvxpy.settings as settings
import numpy as np
import scipy.linalg as la
import scipy.sparse as spar
import scipy.sparse.linalg as sparla
from scipy.sparse import csc_matrix
class SparseCholeskyMessages:
    ASYMMETRIC = 'Input matrix is not symmetric to within provided tolerance.'
    INDEFINITE = 'Input matrix is neither positive nor negative definite.'
    EIGEN_FAIL = 'Cholesky decomposition failed.'
    NOT_SPARSE = 'Input must be a SciPy sparse matrix.'
    NOT_REAL = 'Input matrix must be real.'