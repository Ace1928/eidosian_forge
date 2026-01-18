from __future__ import annotations
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit
from qiskit.synthesis.linear.linear_matrix_utils import (
def _get_label_arr(n, mat_t):
    label_arr = []
    for i in range(n):
        j = 0
        while not mat_t[i, n - 1 - j]:
            j += 1
        label_arr.append(j)
    return label_arr