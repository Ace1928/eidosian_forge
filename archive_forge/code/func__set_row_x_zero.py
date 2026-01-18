import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Clifford
from qiskit.quantum_info.operators.symplectic.clifford_circuits import (
from .clifford_decompose_bm import _decompose_clifford_1q
def _set_row_x_zero(clifford, circuit, qubit):
    """Set destabilizer.X[qubit, i] to False for all i > qubit.

    This is done by applying CNOTs assuming :math:`k \\leq N` and A[k][k]=1
    """
    x = clifford.destab_x[qubit]
    z = clifford.destab_z[qubit]
    for i in range(qubit + 1, clifford.num_qubits):
        if x[i]:
            _append_cx(clifford, qubit, i)
            circuit.cx(qubit, i)
    if np.any(z[qubit:]):
        if not z[qubit]:
            _append_s(clifford, qubit)
            circuit.s(qubit)
        for i in range(qubit + 1, clifford.num_qubits):
            if z[i]:
                _append_cx(clifford, i, qubit)
                circuit.cx(i, qubit)
        _append_s(clifford, qubit)
        circuit.s(qubit)