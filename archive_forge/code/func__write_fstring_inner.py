import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
def _write_fstring_inner(self, node):
    if isinstance(node, JoinedStr):
        for value in node.values:
            self._write_fstring_inner(value)
    elif isinstance(node, Constant) and isinstance(node.value, str):
        value = node.value.replace('{', '{{').replace('}', '}}')
        self.write(value)
    elif isinstance(node, FormattedValue):
        self.visit_FormattedValue(node)
    else:
        raise ValueError(f'Unexpected node inside JoinedStr, {node!r}')