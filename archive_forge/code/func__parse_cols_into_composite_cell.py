import json
import urllib.parse
from typing import (
import numpy as np
from cirq import devices, circuits, ops, protocols
from cirq.interop.quirk.cells import (
from cirq.interop.quirk.cells.parse import parse_matrix
def _parse_cols_into_composite_cell(data: Dict[str, Any], registry: Dict[str, CellMaker]) -> CompositeCell:
    if not isinstance(data, Dict):
        raise ValueError('Circuit JSON must be a dictionary.')
    if 'cols' not in data:
        raise ValueError(f'Circuit JSON dict must have a "cols" entry.\nJSON={data}')
    cols = data['cols']
    if not isinstance(cols, list):
        raise ValueError(f'Circuit JSON cols must be a list.\nJSON={data}')
    parsed_cols: List[List[Optional[Cell]]] = []
    height = 0
    for i, col in enumerate(cols):
        parsed_col, h = _parse_col_cells_with_height(registry, i, col)
        height = max(height, h)
        parsed_cols.append(parsed_col)
    for col in parsed_cols:
        for i in range(len(col)):
            cell = col[i]
            if cell is not None:
                cell.modify_column(col)
    persistent_mods = {}
    for c in parsed_cols:
        for cell in c:
            if cell is not None:
                for key, modifier in cell.persistent_modifiers().items():
                    persistent_mods[key] = modifier
        for i in range(len(c)):
            for modifier in persistent_mods.values():
                cell = c[i]
                if cell is not None:
                    c[i] = modifier(cell)
    gate_count = sum((0 if cell is None else cell.gate_count() for col in parsed_cols for cell in col))
    return CompositeCell(height, parsed_cols, gate_count=gate_count)