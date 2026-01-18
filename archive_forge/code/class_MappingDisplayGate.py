import abc
from typing import (
from cirq import circuits, ops, protocols, transformers, value
from cirq.type_workarounds import NotImplementedType
class MappingDisplayGate(ops.Gate):
    """Displays the indices mapped to a set of wires."""

    def __init__(self, indices):
        self.indices = tuple(indices)
        self._num_qubits = len(self.indices)

    def num_qubits(self) -> int:
        return self._num_qubits

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> 'cirq.CircuitDiagramInfo':
        wire_symbols = tuple(('' if i is None else str(i) for i in self.indices))
        return protocols.CircuitDiagramInfo(wire_symbols, connected=False)