import IPython.display
import numpy as np
import pytest
import cirq
from cirq.contrib.svg import circuit_to_svg
class CustomGate(cirq.Gate):

    def _num_qubits_(self) -> int:
        return 1

    def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=[symbol])