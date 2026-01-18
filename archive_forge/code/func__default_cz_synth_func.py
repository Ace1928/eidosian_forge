from __future__ import annotations
from collections.abc import Callable
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Clifford  # pylint: disable=cyclic-import
from qiskit.quantum_info.operators.symplectic.clifford_circuits import (
from qiskit.synthesis.linear import (
from qiskit.synthesis.linear_phase import synth_cz_depth_line_mr, synth_cx_cz_depth_line_my
from qiskit.synthesis.linear.linear_matrix_utils import (
def _default_cz_synth_func(symmetric_mat):
    """
    Construct the layer of CZ gates from a symmetric matrix.
    """
    nq = symmetric_mat.shape[0]
    qc = QuantumCircuit(nq, name='CZ')
    for j in range(nq):
        for i in range(0, j):
            if symmetric_mat[i][j]:
                qc.cz(i, j)
    return qc