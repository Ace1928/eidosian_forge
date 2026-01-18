import re
from ..helpers import PREVENT_BACKSLASH
def _process_row(text, aligns):
    cells = CELL_SPLIT.split(text)
    if len(cells) != len(aligns):
        return None
    children = [{'type': 'table_cell', 'text': text.strip(), 'attrs': {'align': aligns[i], 'head': False}} for i, text in enumerate(cells)]
    return {'type': 'table_row', 'children': children}