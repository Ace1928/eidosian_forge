from typing import Sequence
import numpy as np
import pytest
import cirq
class UnimplementedUnitaryGate(cirq.testing.TwoQubitGate):

    def _unitary_(self):
        return np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])