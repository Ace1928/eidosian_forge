import pytest
import cirq
class Included(cirq.testing.TwoQubitGate):

    def matrix(self):
        pass