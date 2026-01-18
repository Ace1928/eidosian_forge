import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Clifford
from qiskit.quantum_info.operators.symplectic.clifford_circuits import (
from .clifford_decompose_bm import _decompose_clifford_1q
def _set_qubit_x_true(clifford, circuit, qubit):
    """Set destabilizer.X[qubit, qubit] to be True.

    This is done by permuting columns l > qubit or if necessary applying
    a Hadamard
    """
    x = clifford.destab_x[qubit]
    z = clifford.destab_z[qubit]
    if x[qubit]:
        return
    for i in range(qubit + 1, clifford.num_qubits):
        if x[i]:
            _append_swap(clifford, i, qubit)
            circuit.swap(i, qubit)
            return
    for i in range(qubit, clifford.num_qubits):
        if z[i]:
            _append_h(clifford, i)
            circuit.h(i)
            if i != qubit:
                _append_swap(clifford, i, qubit)
                circuit.swap(i, qubit)
            return