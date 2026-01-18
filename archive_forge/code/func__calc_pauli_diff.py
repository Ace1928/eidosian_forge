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
def _calc_pauli_diff(cliff, cliff_target):
    """Given two Cliffords that differ by a Pauli, we find this Pauli."""
    num_qubits = cliff.num_qubits
    if cliff.num_qubits != cliff_target.num_qubits:
        raise QiskitError('num_qubits is not the same for the original clifford and the target.')
    phase = [cliff.phase[k] ^ cliff_target.phase[k] for k in range(2 * num_qubits)]
    phase = np.array(phase, dtype=int)
    A = cliff.symplectic_matrix
    Ainv = calc_inverse_matrix(A)
    C = np.matmul(Ainv, phase) % 2
    pauli_circ = QuantumCircuit(num_qubits, name='Pauli')
    for k in range(num_qubits):
        destab = C[k]
        stab = C[k + num_qubits]
        if stab and destab:
            pauli_circ.y(k)
        elif stab:
            pauli_circ.x(k)
        elif destab:
            pauli_circ.z(k)
    return pauli_circ