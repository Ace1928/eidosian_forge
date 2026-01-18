import numpy as np
import pytest
import sympy
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
class MinusOnePhaseGate(cirq.testing.SingleQubitGate):
    """Equivalent to a -1 global phase without _act_on_ defined on it."""

    def _unitary_(self):
        return np.array([[-1, 0], [0, -1]])