from typing import Optional, Union
import numpy as np
from qiskit.exceptions import QiskitError
def _gauss_elimination(mat, ncols=None, full_elim=False):
    """Gauss elimination of a matrix mat with m rows and n columns.
    If full_elim = True, it allows full elimination of mat[:, 0 : ncols]
    Returns the matrix mat."""
    mat, _ = _gauss_elimination_with_perm(mat, ncols, full_elim)
    return mat