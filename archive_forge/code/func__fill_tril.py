from __future__ import annotations
import numpy as np
from numpy.random import default_rng
from .clifford import Clifford
from .pauli import Pauli
from .pauli_list import PauliList
def _fill_tril(mat, rng, symmetric=False):
    """Add symmetric random ints to off diagonals"""
    dim = mat.shape[0]
    if dim == 1:
        return
    if dim <= 4:
        mat[1, 0] = rng.integers(2, dtype=np.int8)
        if symmetric:
            mat[0, 1] = mat[1, 0]
        if dim > 2:
            mat[2, 0] = rng.integers(2, dtype=np.int8)
            mat[2, 1] = rng.integers(2, dtype=np.int8)
            if symmetric:
                mat[0, 2] = mat[2, 0]
                mat[1, 2] = mat[2, 1]
        if dim > 3:
            mat[3, 0] = rng.integers(2, dtype=np.int8)
            mat[3, 1] = rng.integers(2, dtype=np.int8)
            mat[3, 2] = rng.integers(2, dtype=np.int8)
            if symmetric:
                mat[0, 3] = mat[3, 0]
                mat[1, 3] = mat[3, 1]
                mat[2, 3] = mat[3, 2]
        return
    rows, cols = np.tril_indices(dim, -1)
    vals = rng.integers(2, size=rows.size, dtype=np.int8)
    mat[rows, cols] = vals
    if symmetric:
        mat[cols, rows] = vals