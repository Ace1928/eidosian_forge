import pytest
import numpy as np
import cirq
class _MixtureGate(cirq.testing.SingleQubitGate):

    def __init__(self, p, q):
        self._p = p
        self._q = q
        super().__init__()

    def _mixture_(self):
        return ((self._p, cirq.unitary(cirq.I)), (self._q, cirq.unitary(cirq.X)))