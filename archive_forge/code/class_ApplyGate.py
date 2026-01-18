from typing import Optional
import numpy as np
import pytest
import cirq
from cirq import testing
class ApplyGate(cirq.Gate):

    def num_qubits(self):
        return 1

    def _apply_unitary_(self, args):
        return cirq.apply_unitary(cirq.X(cirq.LineQubit(0)), args)