from typing import Optional, List, Iterator, Iterable, TYPE_CHECKING
from cirq.interop.quirk.cells.cell import Cell, CELL_SIZES, CellMaker
def _input_family(identifier_prefix: str, letter: str, rev: bool=False) -> Iterator[CellMaker]:
    for n in CELL_SIZES:
        yield CellMaker(identifier=identifier_prefix + str(n), size=n, maker=lambda args: InputCell(qubits=args.qubits[::-1] if rev else args.qubits, letter=letter))