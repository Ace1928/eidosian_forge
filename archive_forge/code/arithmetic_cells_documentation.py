import inspect
from typing import (
from cirq import ops, value
from cirq.interop.quirk.cells.cell import Cell, CellMaker, CELL_SIZES
Inits _QuirkArithmeticCallable.

        Args:
            func: Maps target int to its output value based on other input ints.
        