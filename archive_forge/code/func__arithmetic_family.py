import inspect
from typing import (
from cirq import ops, value
from cirq.interop.quirk.cells.cell import Cell, CellMaker, CELL_SIZES
def _arithmetic_family(identifier_prefix: str, func: _IntsToIntCallable) -> Iterator[CellMaker]:
    yield from _size_dependent_arithmetic_family(identifier_prefix, size_to_func=lambda _: func)