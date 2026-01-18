import itertools
import re
from typing import cast, Tuple, Union
import numpy as np
import pytest
import sympy
import cirq
from cirq import protocols
from cirq.type_workarounds import NotImplementedType
class GateAllocatingNewSpaceForResult(cirq.testing.SingleQubitGate):

    def __init__(self):
        self._matrix = cirq.testing.random_unitary(2, random_state=1234)

    def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> Union[np.ndarray, NotImplementedType]:
        assert len(args.axes) == 1
        a = args.axes[0]
        seed = cast(Tuple[Union[int, slice, 'ellipsis'], ...], (slice(None),))
        zero = seed * a + (0, Ellipsis)
        one = seed * a + (1, Ellipsis)
        result = np.zeros(args.target_tensor.shape, args.target_tensor.dtype)
        result[zero] = args.target_tensor[zero] * self._matrix[0][0] + args.target_tensor[one] * self._matrix[0][1]
        result[one] = args.target_tensor[zero] * self._matrix[1][0] + args.target_tensor[one] * self._matrix[1][1]
        return result

    def _unitary_(self):
        return self._matrix

    def __eq__(self, other):
        return isinstance(other, type(self))

    def __repr__(self):
        return 'cirq.ops.controlled_operation_test.GateAllocatingNewSpaceForResult()'