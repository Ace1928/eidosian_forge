from typing import Any, Iterable, Iterator, List, Optional, TYPE_CHECKING, Union
from cirq import ops, value
from cirq.interop.quirk.cells.cell import Cell, CellMaker
@value.value_equality(unhashable=True)
class ParityControlCell(Cell):
    """A modifier that adds a group parity control to other cells in the column.

    The parity controls in a column are satisfied *as a group* if an odd number
    of them are individually satisfied.
    """

    def __init__(self, qubits: Iterable['cirq.Qid'], basis_change: Iterable['cirq.Operation']):
        self.qubits = list(qubits)
        self._basis_change = list(basis_change)

    def _value_equality_values_(self) -> Any:
        return (self.qubits, self._basis_change)

    def __repr__(self) -> str:
        return f'cirq.interop.quirk.cells.control_cells.ParityControlCell(\n    {self.qubits!r},\n    {self._basis_change!r})'

    def gate_count(self) -> int:
        return 0

    def with_line_qubits_mapped_to(self, qubits: List['cirq.Qid']) -> 'Cell':
        return ParityControlCell(qubits=Cell._replace_qubits(self.qubits, qubits), basis_change=tuple((op.with_qubits(*Cell._replace_qubits(op.qubits, qubits)) for op in self._basis_change)))

    def modify_column(self, column: List[Optional['Cell']]):
        for i in range(len(column)):
            gate = column[i]
            if gate is self:
                continue
            elif isinstance(gate, ParityControlCell):
                column[i] = None
                self._basis_change += gate._basis_change
                self.qubits += gate.qubits
            elif gate is not None:
                column[i] = gate.controlled_by(self.qubits[0])

    def basis_change(self) -> 'cirq.OP_TREE':
        yield from self._basis_change
        for q in self.qubits[1:]:
            yield ops.CNOT(q, self.qubits[0])