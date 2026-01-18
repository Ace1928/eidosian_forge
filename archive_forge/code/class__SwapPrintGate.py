from typing import Dict, Optional, Tuple, TYPE_CHECKING
from cirq import circuits, ops
class _SwapPrintGate(ops.Gate):
    """A gate that displays the string representation of each qubits on the circuit."""

    def __init__(self, qubits: Tuple[Tuple['cirq.Qid', 'cirq.Qid'], ...]) -> None:
        self.qubits = qubits

    def num_qubits(self):
        return len(self.qubits)

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs') -> Tuple[str, ...]:
        return tuple((f'{str(q[1])}' for q in self.qubits))