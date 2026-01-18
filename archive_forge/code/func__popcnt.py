import inspect
from typing import (
from cirq import ops, value
from cirq.interop.quirk.cells.cell import Cell, CellMaker, CELL_SIZES
def _popcnt(a: int) -> int:
    """Returns the Hamming weight of the given non-negative integer."""
    t = 0
    while a > 0:
        a &= a - 1
        t += 1
    return t