from typing import Callable, Iterator, Sequence, Tuple, TYPE_CHECKING
from cirq import ops, value
from cirq.interop.quirk.cells.cell import CELL_SIZES, CellMaker
def _interleave_bit(n: int, x: int) -> int:
    h = (n + 1) // 2
    group = x // h
    stride = x % h
    return stride * 2 + group