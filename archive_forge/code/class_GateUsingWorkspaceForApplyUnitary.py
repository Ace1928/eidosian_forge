import itertools
import re
from typing import cast, Tuple, Union
import numpy as np
import pytest
import sympy
import cirq
from cirq import protocols
from cirq.type_workarounds import NotImplementedType
class GateUsingWorkspaceForApplyUnitary(cirq.testing.SingleQubitGate):

    def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> Union[np.ndarray, NotImplementedType]:
        args.available_buffer[...] = args.target_tensor
        args.target_tensor[...] = 0
        return args.available_buffer

    def _unitary_(self):
        return np.eye(2)

    def __eq__(self, other):
        return isinstance(other, type(self))

    def __repr__(self):
        return 'cirq.ops.controlled_operation_test.GateUsingWorkspaceForApplyUnitary()'