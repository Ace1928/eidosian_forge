from typing import AbstractSet, Sequence, Union, List, Tuple
import pytest
import numpy as np
import sympy
import cirq
from cirq._compat import proper_repr
from cirq.type_workarounds import NotImplementedType
import cirq.testing.consistent_controlled_gate_op_test as controlled_gate_op_test
class BadGateApplyUnitaryToTensor(GoodGate):

    def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> Union[np.ndarray, NotImplementedType]:
        if self.exponent != 1 or cirq.is_parameterized(self):
            return NotImplemented
        zero = cirq.slice_for_qubits_equal_to(args.axes, 0)
        one = cirq.slice_for_qubits_equal_to(args.axes, 1)
        c = np.exp(1j * np.pi * self.phase_exponent)
        args.target_tensor[one] *= c
        args.available_buffer[zero] = args.target_tensor[one]
        args.available_buffer[one] = args.target_tensor[zero]
        args.available_buffer[one] *= c
        return args.available_buffer