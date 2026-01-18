import json
import urllib.parse
from typing import (
import numpy as np
from cirq import devices, circuits, ops, protocols
from cirq.interop.quirk.cells import (
from cirq.interop.quirk.cells.parse import parse_matrix
def _parse_col_cells_with_height(registry: Dict[str, CellMaker], col: int, col_data: Any) -> Tuple[List[Optional[Cell]], int]:
    if not isinstance(col_data, list):
        raise ValueError(f'col must be a list.\ncol: {col_data!r}')
    result = []
    height = 0
    for row in range(len(col_data)):
        cell, h = _parse_cell_with_height(registry, row, col, col_data[row])
        result.append(cell)
        height = max(height, h + row)
    return (result, height)