from __future__ import annotations
import numpy as np
from numpy.random import default_rng
from .clifford import Clifford
from .pauli import Pauli
from .pauli_list import PauliList
def _inverse_tril(mat, block_inverse_threshold):
    """Invert a lower-triangular matrix with unit diagonal."""
    dim = mat.shape[0]
    if dim <= 2:
        return mat
    if dim <= 5:
        inv = mat.copy()
        inv[2, 0] = mat[2, 0] ^ mat[1, 0] & mat[2, 1]
        if dim > 3:
            inv[3, 1] = mat[3, 1] ^ mat[2, 1] & mat[3, 2]
            inv[3, 0] = mat[3, 0] ^ mat[3, 2] & mat[2, 0] ^ mat[1, 0] & inv[3, 1]
        if dim > 4:
            inv[4, 2] = (mat[4, 2] ^ mat[3, 2] & mat[4, 3]) & 1
            inv[4, 1] = mat[4, 1] ^ mat[4, 3] & mat[3, 1] ^ mat[2, 1] & inv[4, 2]
            inv[4, 0] = mat[4, 0] ^ mat[1, 0] & inv[4, 1] ^ mat[2, 0] & inv[4, 2] ^ mat[3, 0] & mat[4, 3]
        return inv % 2
    if dim <= block_inverse_threshold:
        return np.linalg.inv(mat).astype(np.int8) % 2
    dim1 = dim // 2
    mat_a = _inverse_tril(mat[0:dim1, 0:dim1], block_inverse_threshold)
    mat_d = _inverse_tril(mat[dim1:dim, dim1:dim], block_inverse_threshold)
    mat_c = np.matmul(np.matmul(mat_d, mat[dim1:dim, 0:dim1]), mat_a)
    inv = np.block([[mat_a, np.zeros((dim1, dim - dim1), dtype=int)], [mat_c, mat_d]])
    return inv % 2