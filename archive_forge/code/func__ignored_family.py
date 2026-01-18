from typing import Iterator
from cirq.interop.quirk.cells.cell import CellMaker, CELL_SIZES
def _ignored_family(identifier_prefix: str) -> Iterator[CellMaker]:
    yield _ignored_gate(identifier_prefix)
    for i in CELL_SIZES:
        yield _ignored_gate(identifier_prefix + str(i))