from __future__ import annotations
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit
from qiskit.synthesis.linear.linear_matrix_utils import (
def _get_label_arr_t(n, label_arr):
    label_arr_t = [None] * n
    for i in range(n):
        label_arr_t[label_arr[i]] = i
    return label_arr_t