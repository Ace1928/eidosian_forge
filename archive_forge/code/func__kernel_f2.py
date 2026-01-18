from __future__ import annotations
import itertools
from collections.abc import Iterable
from copy import deepcopy
from typing import Union, cast
import numpy as np
from qiskit.exceptions import QiskitError
from ..operators import Pauli, SparsePauliOp
def _kernel_f2(matrix_in):
    """
    Compute the kernel of a binary matrix on the binary finite field.

    Args:
        matrix_in (numpy.ndarray): Binary matrix.

    Returns:
        The list of kernel vectors.
    """
    size = matrix_in.shape
    kernel = []
    matrix_in_id = np.vstack((matrix_in, np.identity(size[1])))
    matrix_in_id_ech = _row_echelon_f2(matrix_in_id.transpose()).transpose()
    for col in range(size[1]):
        if np.array_equal(matrix_in_id_ech[0:size[0], col], np.zeros(size[0])) and (not np.array_equal(matrix_in_id_ech[size[0]:, col], np.zeros(size[1]))):
            kernel.append(matrix_in_id_ech[size[0]:, col])
    return kernel