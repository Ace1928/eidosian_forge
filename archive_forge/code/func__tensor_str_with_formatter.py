import contextlib
import dataclasses
import math
import textwrap
from typing import Any, Dict, Optional
import torch
from torch import inf
def _tensor_str_with_formatter(self, indent, summarize, formatter1, formatter2=None):
    dim = self.dim()
    if dim == 0:
        return _scalar_str(self, formatter1, formatter2)
    if dim == 1:
        return _vector_str(self, indent, summarize, formatter1, formatter2)
    if summarize and self.size(0) > 2 * PRINT_OPTS.edgeitems:
        slices = [_tensor_str_with_formatter(self[i], indent + 1, summarize, formatter1, formatter2) for i in range(0, PRINT_OPTS.edgeitems)] + ['...'] + [_tensor_str_with_formatter(self[i], indent + 1, summarize, formatter1, formatter2) for i in range(len(self) - PRINT_OPTS.edgeitems, len(self))]
    else:
        slices = [_tensor_str_with_formatter(self[i], indent + 1, summarize, formatter1, formatter2) for i in range(0, self.size(0))]
    tensor_str = (',' + '\n' * (dim - 1) + ' ' * (indent + 1)).join(slices)
    return '[' + tensor_str + ']'