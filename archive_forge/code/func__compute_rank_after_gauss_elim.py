from typing import Optional, Union
import numpy as np
from qiskit.exceptions import QiskitError
def _compute_rank_after_gauss_elim(mat):
    """Given a matrix A after Gaussian elimination, computes its rank
    (i.e. simply the number of nonzero rows)"""
    return np.sum(mat.any(axis=1))