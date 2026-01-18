from typing import AbstractSet, Sequence, Union, List, Tuple
import pytest
import numpy as np
import sympy
import cirq
from cirq._compat import proper_repr
from cirq.type_workarounds import NotImplementedType
import cirq.testing.consistent_controlled_gate_op_test as controlled_gate_op_test
class BadGateIsParameterized(GoodGate):

    def _is_parameterized_(self) -> bool:
        return not super()._is_parameterized_()