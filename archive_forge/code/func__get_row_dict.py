import re
from itertools import chain
from ._base import (
from .exonerate_vulgar import _RE_VULGAR
def _get_row_dict(row_len, model):
    """Return a dictionary of row indices for parsing alignment blocks (PRIVATE)."""
    idx = {}
    if row_len == 3:
        idx['query'] = 0
        idx['midline'] = 1
        idx['hit'] = 2
        idx['qannot'], idx['hannot'] = (None, None)
    elif row_len == 4:
        if 'protein2' in model:
            idx['query'] = 0
            idx['midline'] = 1
            idx['hit'] = 2
            idx['hannot'] = 3
            idx['qannot'] = None
        elif '2protein' in model:
            idx['query'] = 1
            idx['midline'] = 2
            idx['hit'] = 3
            idx['hannot'] = None
            idx['qannot'] = 0
        else:
            raise ValueError('Unexpected model: ' + model)
    elif row_len == 5:
        idx['qannot'] = 0
        idx['query'] = 1
        idx['midline'] = 2
        idx['hit'] = 3
        idx['hannot'] = 4
    else:
        raise ValueError('Unexpected row count in alignment block: %i' % row_len)
    return idx