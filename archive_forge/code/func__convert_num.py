import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
def _convert_num(node):
    if not isinstance(node, Constant) or type(node.value) not in (int, float, complex):
        _raise_malformed_node(node)
    return node.value