from typing import Iterator
from cirq.interop.quirk.cells.cell import CellMaker, CELL_SIZES
def _unsupported_family(identifier_prefix: str, reason: str) -> Iterator[CellMaker]:
    for i in CELL_SIZES:
        yield _unsupported_gate(identifier_prefix + str(i), reason)