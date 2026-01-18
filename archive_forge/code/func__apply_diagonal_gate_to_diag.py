from __future__ import annotations
import itertools
import numpy as np
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_isometry
from .diagonal import Diagonal
from .uc import UCGate
from .mcg_up_to_diagonal import MCGupDiag
def _apply_diagonal_gate_to_diag(m_diagonal, action_qubit_labels, diag, num_qubits):
    if not m_diagonal:
        return m_diagonal
    basis_states = list(itertools.product([0, 1], repeat=num_qubits))
    for state in basis_states[:len(m_diagonal)]:
        state_on_action_qubits = [state[i] for i in action_qubit_labels]
        diag_index = _bin_to_int(state_on_action_qubits)
        i = _bin_to_int(state)
        m_diagonal[i] *= diag[diag_index]
    return m_diagonal