import pytest
import cirq
from cirq.interop.quirk.cells.cell import Cell, ExplicitOperationsCell
class BasicCell(Cell):

    def with_line_qubits_mapped_to(self, qubits):
        raise NotImplementedError()

    def gate_count(self) -> int:
        raise NotImplementedError()