from __future__ import annotations
import itertools
from collections.abc import Iterable
from copy import deepcopy
from typing import Union, cast
import numpy as np
from qiskit.exceptions import QiskitError
from ..operators import Pauli, SparsePauliOp
def _row_echelon_f2(matrix_in):
    """
    Compute the row Echelon form of a binary matrix on the binary finite field.

    Args:
        matrix_in (numpy.ndarray): Binary matrix.

    Returns:
        Matrix_in in Echelon row form.
    """
    size = matrix_in.shape
    for i in range(size[0]):
        pivot_index = 0
        for j in range(size[1]):
            if matrix_in[i, j] == 1:
                pivot_index = j
                break
        for k in range(size[0]):
            if k != i and matrix_in[k, pivot_index] == 1:
                matrix_in[k, :] = np.mod(matrix_in[k, :] + matrix_in[i, :], 2)
    matrix_out_temp = deepcopy(matrix_in)
    indices = []
    matrix_out = np.zeros(size)
    for i in range(size[0] - 1):
        if np.array_equal(matrix_out_temp[i, :], np.zeros(size[1])):
            indices.append(i)
    for row in np.sort(indices)[::-1]:
        matrix_out_temp = np.delete(matrix_out_temp, row, axis=0)
    matrix_out[0:size[0] - len(indices), :] = matrix_out_temp
    matrix_out = matrix_out.astype(int)
    return matrix_out