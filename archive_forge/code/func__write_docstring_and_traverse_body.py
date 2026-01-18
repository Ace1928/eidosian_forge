import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
def _write_docstring_and_traverse_body(self, node):
    if (docstring := self.get_raw_docstring(node)):
        self._write_docstring(docstring)
        self.traverse(node.body[1:])
    else:
        self.traverse(node.body)