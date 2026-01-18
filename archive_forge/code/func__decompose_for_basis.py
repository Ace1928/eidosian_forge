from typing import (
import numpy as np
import sympy
from cirq import protocols, value
from cirq._compat import proper_repr
from cirq.ops import common_gates, raw_types, global_phase_op
def _decompose_for_basis(self, index: int, bit_flip: int, theta: 'cirq.TParamVal', qubits: Sequence['cirq.Qid']) -> Iterator[Union['cirq.ZPowGate', 'cirq.CXPowGate']]:
    if index == 0:
        return []
    largest_digit = self._num_qubits_() - (len(bin(index)) - 2)
    yield common_gates.rz(2 * theta)(qubits[largest_digit])
    _flip_bit = self._num_qubits_() - bit_flip - 1
    if _flip_bit < largest_digit:
        yield common_gates.CNOT(qubits[largest_digit], qubits[_flip_bit])
    elif _flip_bit > largest_digit:
        yield common_gates.CNOT(qubits[_flip_bit], qubits[largest_digit])