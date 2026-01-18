from typing import AbstractSet, Sequence, Union, List, Tuple
import pytest
import numpy as np
import sympy
import cirq
from cirq._compat import proper_repr
from cirq.type_workarounds import NotImplementedType
import cirq.testing.consistent_controlled_gate_op_test as controlled_gate_op_test
class BadGateRepr(GoodGate):

    def __repr__(self):
        args = [f'phase_exponent={2 * self.phase_exponent!r}']
        if self.exponent != 1:
            args.append(f'exponent={proper_repr(self.exponent)}')
        return f'BadGateRepr({', '.join(args)})'