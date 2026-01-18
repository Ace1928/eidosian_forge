from __future__ import annotations
from collections.abc import Sequence
import numpy as np
from qiskit.circuit.library.pauli_evolution import PauliEvolutionGate
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.quantum_info import Operator, Pauli, SparsePauliOp
from qiskit.synthesis.evolution import LieTrotter
from .n_local import NLocal
def _is_pauli_identity(operator):
    if isinstance(operator, SparsePauliOp):
        if len(operator.paulis) == 1:
            operator = operator.paulis[0]
        else:
            return False
    if isinstance(operator, Pauli):
        return not np.any(np.logical_or(operator.x, operator.z))
    return False