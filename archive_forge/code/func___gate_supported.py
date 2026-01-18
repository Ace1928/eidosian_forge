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
def __gate_supported(self, gate: Gate, qarg: int) -> bool:
    """A gate is supported on the qubit (qarg) or not."""
    if self.target is None or self.target.instruction_supported(gate.name, qargs=(qarg,)):
        return True
    return False