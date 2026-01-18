from typing import AbstractSet, Sequence, Union, List, Tuple
import pytest
import numpy as np
import sympy
import cirq
from cirq._compat import proper_repr
from cirq.type_workarounds import NotImplementedType
import cirq.testing.consistent_controlled_gate_op_test as controlled_gate_op_test
class BadGatePhaseBy(GoodGate):

    def _phase_by_(self, phase_turns, qubit_index):
        assert qubit_index == 0
        return BadGatePhaseBy(exponent=self.exponent, phase_exponent=self.phase_exponent + phase_turns * 4)