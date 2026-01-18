from __future__ import annotations
import copy
from itertools import zip_longest
import math
from typing import List, Type
import numpy
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.qobj.qasm_qobj import QasmQobjInstruction
from qiskit.circuit.parameter import ParameterExpression
from qiskit.circuit.operation import Operation
from qiskit.circuit.annotated_operation import AnnotatedOperation, InverseModifier
def _qasmif(self, string):
    """Print an if statement if needed."""
    from qiskit.qasm2 import QASM2ExportError
    if self.condition is None:
        return string
    if not isinstance(self.condition[0], ClassicalRegister):
        raise QASM2ExportError("OpenQASM 2 can only condition on registers, but got '{self.condition[0]}'")
    return 'if(%s==%d) ' % (self.condition[0].name, self.condition[1]) + string