from typing import AbstractSet, Sequence, Union, List, Tuple
import pytest
import numpy as np
import sympy
import cirq
from cirq._compat import proper_repr
from cirq.type_workarounds import NotImplementedType
import cirq.testing.consistent_controlled_gate_op_test as controlled_gate_op_test
class BadGatePauliExpansion(GoodGate):

    def _pauli_expansion_(self) -> cirq.LinearDict[str]:
        return cirq.LinearDict({'I': 10})