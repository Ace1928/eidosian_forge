from typing import Iterator
from cirq.interop.quirk.cells.cell import CellMaker, CELL_SIZES
def _ignored_gate(identifier: str) -> CellMaker:
    return CellMaker(identifier, size=0, maker=lambda _: None)