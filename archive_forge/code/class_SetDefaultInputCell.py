from typing import Optional, List, Iterator, Iterable, TYPE_CHECKING
from cirq.interop.quirk.cells.cell import Cell, CELL_SIZES, CellMaker
class SetDefaultInputCell(Cell):
    """A persistent modifier that provides a fallback classical input."""

    def __init__(self, letter: str, value: int):
        self.letter = letter
        self.value = value

    def gate_count(self) -> int:
        return 0

    def with_line_qubits_mapped_to(self, qubits: List['cirq.Qid']) -> 'Cell':
        return self

    def persistent_modifiers(self):
        return {f'set_default_{self.letter}': lambda cell: cell.with_input(self.letter, self.value)}