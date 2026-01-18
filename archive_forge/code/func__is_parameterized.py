from __future__ import annotations
from qiskit.circuit import Instruction, ParameterExpression, Qubit, Clbit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.circuit.equivalence_library import EquivalenceLibrary
from qiskit.exceptions import QiskitError
from qiskit.transpiler import Target
from qiskit.transpiler.basepasses import TransformationPass
from .basis_translator import BasisTranslator
def _is_parameterized(op: Instruction) -> bool:
    return any((isinstance(param, ParameterExpression) and len(param.parameters) > 0 for param in op.params))