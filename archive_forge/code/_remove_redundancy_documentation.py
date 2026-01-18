import numpy as np
from scipy.linalg import svd
from scipy.linalg.interpolative import interp_decomp
import scipy
from scipy.linalg.blas import dtrsm
Eliminates redundant equations from a system of equations.

    Eliminates redundant equations from system of equations defined by Ax = b
    and identifies infeasibilities.

    Parameters
    ----------
    A : 2-D array
        An array representing the left-hand side of a system of equations
    rhs : 1-D array
        An array representing the right-hand side of a system of equations
    rank : int, optional
        The rank of A
    randomized: bool, optional
        True for randomized interpolative decomposition

    Returns
    -------
    A : 2-D array
        An array representing the left-hand side of a system of equations
    rhs : 1-D array
        An array representing the right-hand side of a system of equations
    status: int
        An integer indicating the status of the system
        0: No infeasibility identified
        2: Trivially infeasible
    message : str
        A string descriptor of the exit status of the optimization.

    