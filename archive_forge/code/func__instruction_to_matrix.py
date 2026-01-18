from __future__ import annotations
import copy
import re
from numbers import Number
from typing import TYPE_CHECKING
import numpy as np
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.library.standard_gates import HGate, IGate, SGate, TGate, XGate, YGate, ZGate
from qiskit.circuit.operation import Operation
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.linear_op import LinearOp
from qiskit.quantum_info.operators.mixins import generate_apidocs
from qiskit.quantum_info.operators.predicates import is_unitary_matrix, matrix_equal
@classmethod
def _instruction_to_matrix(cls, obj):
    """Return Operator for instruction if defined or None otherwise."""
    from qiskit.quantum_info import Clifford
    from qiskit.circuit.annotated_operation import AnnotatedOperation
    if not isinstance(obj, (Instruction, Clifford, AnnotatedOperation)):
        raise QiskitError('Input is neither Instruction, Clifford or AnnotatedOperation.')
    mat = None
    if hasattr(obj, 'to_matrix'):
        try:
            mat = obj.to_matrix()
        except QiskitError:
            pass
    return mat