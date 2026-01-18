import inspect
from typing import (
from cirq import ops, value
from cirq.interop.quirk.cells.cell import Cell, CellMaker, CELL_SIZES
def _indented_list_lines_repr(items: Sequence[Any]) -> str:
    block = '\n'.join([repr(op) + ',' for op in items])
    indented = '        ' + '\n        '.join(block.split('\n'))
    return f'[\n{indented}\n    ]'