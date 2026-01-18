from __future__ import annotations
import numpy as np
from qiskit.circuit.parametervector import ParameterVector
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.quantum_info import SparsePauliOp
from .evolved_operator_ansatz import EvolvedOperatorAnsatz, _is_pauli_identity
def _check_configuration(self, raise_on_failure: bool=True) -> bool:
    """Check if the current configuration is valid."""
    valid = True
    if not super()._check_configuration(raise_on_failure):
        return False
    if self.cost_operator is None:
        valid = False
        if raise_on_failure:
            raise ValueError('The operator representing the cost of the optimization problem is not set')
    if self.initial_state is not None and self.initial_state.num_qubits != self.num_qubits:
        valid = False
        if raise_on_failure:
            raise ValueError('The number of qubits of the initial state {} does not match the number of qubits of the cost operator {}'.format(self.initial_state.num_qubits, self.num_qubits))
    if self.mixer_operator is not None and self.mixer_operator.num_qubits != self.num_qubits:
        valid = False
        if raise_on_failure:
            raise ValueError('The number of qubits of the mixer {} does not match the number of qubits of the cost operator {}'.format(self.mixer_operator.num_qubits, self.num_qubits))
    return valid