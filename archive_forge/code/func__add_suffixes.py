import contextlib
import dataclasses
import math
import textwrap
from typing import Any, Dict, Optional
import torch
from torch import inf
def _add_suffixes(tensor_str, suffixes, indent, force_newline):
    tensor_strs = [tensor_str]
    last_line_len = len(tensor_str) - tensor_str.rfind('\n') + 1
    for suffix in suffixes:
        suffix_len = len(suffix)
        if force_newline or last_line_len + suffix_len + 2 > PRINT_OPTS.linewidth:
            tensor_strs.append(',\n' + ' ' * indent + suffix)
            last_line_len = indent + suffix_len
            force_newline = False
        else:
            tensor_strs.append(', ' + suffix)
            last_line_len += suffix_len + 2
    tensor_strs.append(')')
    return ''.join(tensor_strs)