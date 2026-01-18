import pytest
import cirq
class DiagramGate(cirq.PauliStringGateOperation):

    def map_qubits(self, qubit_map):
        pass

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        return self._pauli_string_diagram_info(args)