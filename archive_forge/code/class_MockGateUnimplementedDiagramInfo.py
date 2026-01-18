import cirq
import cirq_web
import pytest
class MockGateUnimplementedDiagramInfo(cirq.testing.SingleQubitGate):

    def __init__(self):
        super(MockGateUnimplementedDiagramInfo, self)

    def _circuit_diagram_info_(self, args):
        return NotImplemented