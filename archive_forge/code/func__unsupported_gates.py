from typing import Iterator
from cirq.interop.quirk.cells.cell import CellMaker, CELL_SIZES
def _unsupported_gates(*identifiers: str, reason: str) -> Iterator[CellMaker]:
    for identifier in identifiers:
        yield _unsupported_gate(identifier, reason)