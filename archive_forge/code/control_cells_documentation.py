from typing import Any, Iterable, Iterator, List, Optional, TYPE_CHECKING, Union
from cirq import ops, value
from cirq.interop.quirk.cells.cell import Cell, CellMaker
A modifier that adds a group parity control to other cells in the column.

    The parity controls in a column are satisfied *as a group* if an odd number
    of them are individually satisfied.
    