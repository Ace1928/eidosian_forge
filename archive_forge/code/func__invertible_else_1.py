import inspect
from typing import (
from cirq import ops, value
from cirq.interop.quirk.cells.cell import Cell, CellMaker, CELL_SIZES
def _invertible_else_1(a: int, m: int) -> int:
    """Returns `a` if `a` has a multiplicative inverse, else 1."""
    i = _mod_inv_else_1(a, m)
    return a if i != 1 else i