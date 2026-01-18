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
def _validate_prefix(parameter_prefix, operators):
    if isinstance(parameter_prefix, str):
        return len(operators) * [parameter_prefix]
    if len(parameter_prefix) != len(operators):
        raise ValueError('The number of parameter prefixes must match the operators.')
    return parameter_prefix