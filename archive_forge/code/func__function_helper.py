import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
def _function_helper(self, node, fill_suffix):
    self.maybe_newline()
    for deco in node.decorator_list:
        self.fill('@')
        self.traverse(deco)
    def_str = fill_suffix + ' ' + node.name
    self.fill(def_str)
    with self.delimit('(', ')'):
        self.traverse(node.args)
    if node.returns:
        self.write(' -> ')
        self.traverse(node.returns)
    with self.block(extra=self.get_type_comment(node)):
        self._write_docstring_and_traverse_body(node)