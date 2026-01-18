import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
def _raise_malformed_node(node):
    msg = 'malformed node or string'
    if (lno := getattr(node, 'lineno', None)):
        msg += f' on line {lno}'
    raise ValueError(msg + f': {node!r}')