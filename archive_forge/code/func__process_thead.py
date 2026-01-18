import re
from ..helpers import PREVENT_BACKSLASH
def _process_thead(header, align):
    headers = CELL_SPLIT.split(header)
    aligns = CELL_SPLIT.split(align)
    if len(headers) != len(aligns):
        return (None, None)
    for i, v in enumerate(aligns):
        if ALIGN_CENTER.match(v):
            aligns[i] = 'center'
        elif ALIGN_LEFT.match(v):
            aligns[i] = 'left'
        elif ALIGN_RIGHT.match(v):
            aligns[i] = 'right'
        else:
            aligns[i] = None
    children = [{'type': 'table_cell', 'text': text.strip(), 'attrs': {'align': aligns[i], 'head': True}} for i, text in enumerate(headers)]
    thead = {'type': 'table_head', 'children': children}
    return (thead, aligns)