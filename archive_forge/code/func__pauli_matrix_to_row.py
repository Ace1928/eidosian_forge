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
@staticmethod
def _pauli_matrix_to_row(mat, num_qubits):
    """Generate a binary vector (a row of tableau representation) from a Pauli matrix.
        Return None if the non-Pauli matrix is supplied."""
    decimals = 6

    def find_one_index(x):
        indices = np.where(np.round(np.abs(x), decimals=decimals) == 1)
        return indices[0][0] if len(indices[0]) == 1 else None

    def bitvector(n, num_bits):
        return np.array([int(digit) for digit in format(n, f'0{num_bits}b')], dtype=bool)[::-1]
    xint = find_one_index(mat[0, :])
    if xint is None:
        return None
    xbits = bitvector(xint, num_qubits)
    entries = np.empty(len(mat), dtype=complex)
    for i, row in enumerate(mat):
        index = find_one_index(row)
        if index is None:
            return None
        expected = xint ^ i
        if index != expected:
            return None
        entries[i] = np.round(mat[i, index], decimals=decimals)
        if entries[i] not in {1, -1, 1j, -1j}:
            return None
    zbits = np.empty(num_qubits, dtype=bool)
    for k in range(num_qubits):
        sign = np.round(entries[2 ** k] / entries[0])
        if sign == 1:
            zbits[k] = False
        elif sign == -1:
            zbits[k] = True
        else:
            return None
    phase = None
    num_y = sum(xbits & zbits)
    positive_phase = (-1j) ** num_y
    if entries[0] == positive_phase:
        phase = False
    elif entries[0] == -1 * positive_phase:
        phase = True
    if phase is None:
        return None
    coef = (-1) ** phase * positive_phase
    ivec, zvec = (np.ones(2), np.array([1, -1]))
    expected = coef * functools.reduce(np.kron, [zvec if z else ivec for z in zbits[::-1]])
    if not np.allclose(entries, expected):
        return None
    return np.hstack([xbits, zbits, phase])