from __future__ import annotations
from collections.abc import Collection
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.symplectic import Clifford, Pauli, PauliList
from qiskit.quantum_info.operators.symplectic.clifford_circuits import _append_x
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.circuit import QuantumCircuit, Instruction
def _get_probablities(self, qubits, outcome, outcome_prob, probs):
    """Recursive helper function for calculating the probabilities"""
    qubit_for_branching = -1
    ret = self.copy()
    for i in range(len(qubits)):
        qubit = qubits[len(qubits) - i - 1]
        if outcome[i] == 'X':
            is_deterministic = not any(ret.clifford.stab_x[:, qubit])
            if is_deterministic:
                single_qubit_outcome = ret._measure_and_update(qubit, 0)
                if single_qubit_outcome:
                    outcome[i] = '1'
                else:
                    outcome[i] = '0'
            else:
                qubit_for_branching = i
    if qubit_for_branching == -1:
        str_outcome = ''.join(outcome)
        probs[str_outcome] = outcome_prob
        return
    for single_qubit_outcome in range(0, 2):
        new_outcome = outcome.copy()
        if single_qubit_outcome:
            new_outcome[qubit_for_branching] = '1'
        else:
            new_outcome[qubit_for_branching] = '0'
        stab_cpy = ret.copy()
        stab_cpy._measure_and_update(qubits[len(qubits) - qubit_for_branching - 1], single_qubit_outcome)
        stab_cpy._get_probablities(qubits, new_outcome, 0.5 * outcome_prob, probs)