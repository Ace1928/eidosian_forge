from __future__ import annotations
import functools
import itertools
import re
from typing import Literal
import numpy as np
from qiskit.circuit import Instruction, QuantumCircuit
from qiskit.circuit.library.standard_gates import HGate, IGate, SGate, XGate, YGate, ZGate
from qiskit.circuit.operation import Operation
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.mixins import AdjointMixin, generate_apidocs
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.scalar_op import ScalarOp
from qiskit.quantum_info.operators.symplectic.base_pauli import _count_y
from .base_pauli import BasePauli
from .clifford_circuits import _append_circuit, _append_operation
@classmethod
def _compose_1q(cls, first, second):
    if cls._COMPOSE_1Q_LOOKUP is None:
        tables_1q = np.array([[[False, True], [True, False]], [[False, True], [True, True]], [[True, False], [False, True]], [[True, False], [True, True]], [[True, True], [False, True]], [[True, True], [True, False]]])
        phases_1q = np.array([[False, False], [False, True], [True, False], [True, True]])
        cliffords = [cls(cls._stack_table_phase(table, phase), validate=False, copy=False) for table, phase in itertools.product(tables_1q, phases_1q)]
        cls._COMPOSE_1Q_LOOKUP = {(cls._hash(left), cls._hash(right)): cls._compose_general(left, right) for left, right in itertools.product(cliffords, repeat=2)}
    return cls._COMPOSE_1Q_LOOKUP[cls._hash(first), cls._hash(second)].copy()