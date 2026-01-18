import contextlib
import dataclasses
import math
import textwrap
from typing import Any, Dict, Optional
import torch
from torch import inf
def _vector_str(self, indent, summarize, formatter1, formatter2=None):
    element_length = formatter1.width() + 2
    if formatter2 is not None:
        element_length += formatter2.width() + 1
    elements_per_line = max(1, int(math.floor((PRINT_OPTS.linewidth - indent) / element_length)))

    def _val_formatter(val, formatter1=formatter1, formatter2=formatter2):
        if formatter2 is not None:
            real_str = formatter1.format(val.real)
            imag_str = (formatter2.format(val.imag) + 'j').lstrip()
            if imag_str[0] == '+' or imag_str[0] == '-':
                return real_str + imag_str
            else:
                return real_str + '+' + imag_str
        else:
            return formatter1.format(val)
    if summarize and (not PRINT_OPTS.edgeitems):
        data = ['...']
    elif summarize and self.size(0) > 2 * PRINT_OPTS.edgeitems:
        data = [_val_formatter(val) for val in self[:PRINT_OPTS.edgeitems].tolist()] + [' ...'] + [_val_formatter(val) for val in self[-PRINT_OPTS.edgeitems:].tolist()]
    else:
        data = [_val_formatter(val) for val in self.tolist()]
    data_lines = [data[i:i + elements_per_line] for i in range(0, len(data), elements_per_line)]
    lines = [', '.join(line) for line in data_lines]
    return '[' + (',' + '\n' + ' ' * (indent + 1)).join(lines) + ']'