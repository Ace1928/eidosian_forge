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
def _compose_general(cls, first, second):
    ifacts = np.sum(second.x & second.z, axis=1, dtype=int)
    x1, z1 = (first.x.astype(np.uint8), first.z.astype(np.uint8))
    lookup = cls._compose_lookup()
    for k, row2 in enumerate(second.symplectic_matrix):
        x1_select = x1[row2]
        z1_select = z1[row2]
        x1_accum = np.logical_xor.accumulate(x1_select, axis=0).astype(np.uint8)
        z1_accum = np.logical_xor.accumulate(z1_select, axis=0).astype(np.uint8)
        indexer = (x1_select[1:], z1_select[1:], x1_accum[:-1], z1_accum[:-1])
        ifacts[k] += np.sum(lookup[indexer])
    p = np.mod(ifacts, 4) // 2
    phase = ((np.matmul(second.symplectic_matrix, first.phase, dtype=int) + second.phase + p) % 2).astype(bool)
    data = cls._stack_table_phase((np.matmul(second.symplectic_matrix, first.symplectic_matrix, dtype=int) % 2).astype(bool), phase)
    return Clifford(data, validate=False, copy=False)