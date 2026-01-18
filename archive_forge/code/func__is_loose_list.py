import re
from .core import BlockState
from .util import (
def _is_loose_list(tokens):
    paragraph_count = 0
    for tok in tokens:
        if tok['type'] == 'blank_line':
            return True
        if tok['type'] == 'paragraph':
            paragraph_count += 1
            if paragraph_count > 1:
                return True