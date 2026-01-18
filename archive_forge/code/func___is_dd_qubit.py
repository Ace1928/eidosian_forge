from __future__ import annotations
import logging
import numpy as np
from qiskit.circuit import Gate, ParameterExpression, Qubit
from qiskit.circuit.delay import Delay
from qiskit.circuit.library.standard_gates import IGate, UGate, U3Gate
from qiskit.circuit.reset import Reset
from qiskit.dagcircuit import DAGCircuit, DAGNode, DAGInNode, DAGOpNode
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.synthesis.one_qubit import OneQubitEulerDecomposer
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.passes.optimization import Optimize1qGates
from qiskit.transpiler.target import Target
from .base_padding import BasePadding
def __is_dd_qubit(self, qubit_index: int) -> bool:
    """DD can be inserted in the qubit or not."""
    if qubit_index in self._no_dd_qubits or (self._qubits and qubit_index not in self._qubits):
        return False
    return True