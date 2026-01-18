import cirq
import cirq_web
import pytest
class MockGateNoDiagramInfo(cirq.testing.SingleQubitGate):

    def __init__(self):
        super(MockGateNoDiagramInfo, self)