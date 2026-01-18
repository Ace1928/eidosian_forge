from __future__ import annotations
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit
from qiskit.synthesis.linear.linear_matrix_utils import (
def _in_linear_combination(label_arr_t, mat_inv_t, row, k):
    indx_k = label_arr_t[k]
    w_needed = np.zeros(len(row), dtype=bool)
    for row_l, _ in enumerate(row):
        if row[row_l]:
            w_needed = w_needed ^ mat_inv_t[row_l]
    if w_needed[indx_k]:
        return False
    return True