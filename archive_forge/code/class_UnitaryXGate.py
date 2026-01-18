from typing import Tuple
import numpy as np
import pytest
import cirq
class UnitaryXGate(cirq.testing.SingleQubitGate):

    def _unitary_(self):
        return np.array([[0, 1], [1, 0]])