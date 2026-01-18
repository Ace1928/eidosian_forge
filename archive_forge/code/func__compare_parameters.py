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
def _compare_parameters(self, other):
    for x, y in zip(self.params, other.params):
        try:
            if not math.isclose(x, y, rel_tol=0, abs_tol=1e-10):
                return False
        except TypeError:
            if x != y:
                return False
    return True