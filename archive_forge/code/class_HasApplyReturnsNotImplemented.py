import numpy as np
import pytest
import cirq
from cirq.protocols.apply_unitary_protocol import _incorporate_result_into_target
class HasApplyReturnsNotImplemented:

    def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs):
        return NotImplemented