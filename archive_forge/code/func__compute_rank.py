from typing import Optional, Union
import numpy as np
from qiskit.exceptions import QiskitError
def _compute_rank(mat):
    """Given a matrix A computes its rank"""
    mat = _gauss_elimination(mat)
    return np.sum(mat.any(axis=1))