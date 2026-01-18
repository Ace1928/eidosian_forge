import numpy as np
import scipy.sparse as sps
from warnings import warn
from ._optimize import OptimizeWarning
from scipy.optimize._remove_redundancy import (
from collections import namedtuple
def _format_A_constraints(A, n_x, sparse_lhs=False):
    """Format the left hand side of the constraints to a 2-D array

    Parameters
    ----------
    A : 2-D array
        2-D array such that ``A @ x`` gives the values of the upper-bound
        (in)equality constraints at ``x``.
    n_x : int
        The number of variables in the linear programming problem.
    sparse_lhs : bool
        Whether either of `A_ub` or `A_eq` are sparse. If true return a
        coo_matrix instead of a numpy array.

    Returns
    -------
    np.ndarray or sparse.coo_matrix
        2-D array such that ``A @ x`` gives the values of the upper-bound
        (in)equality constraints at ``x``.

    """
    if sparse_lhs:
        return sps.coo_matrix((0, n_x) if A is None else A, dtype=float, copy=True)
    elif A is None:
        return np.zeros((0, n_x), dtype=float)
    else:
        return np.array(A, dtype=float, copy=True)