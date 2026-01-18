import dis
import inspect
import sys
from collections import namedtuple
from _pydev_bundle import pydev_log
from opcode import (EXTENDED_ARG, HAVE_ARGUMENT, cmp_op, hascompare, hasconst,
from io import StringIO
import ast as ast_module
class _Visitor(ast_module.NodeVisitor):

    def __init__(self):
        self.try_except_infos = []
        self._stack = []
        self._in_except_stack = []
        self.max_line = -1

    def generic_visit(self, node):
        if hasattr(node, 'lineno'):
            if node.lineno > self.max_line:
                self.max_line = node.lineno
        return ast_module.NodeVisitor.generic_visit(self, node)

    def visit_Try(self, node):
        info = TryExceptInfo(node.lineno, ignore=True)
        self._stack.append(info)
        self.generic_visit(node)
        assert info is self._stack.pop()
        if not info.ignore:
            self.try_except_infos.insert(0, info)
    if sys.version_info[0] < 3:
        visit_TryExcept = visit_Try

    def visit_ExceptHandler(self, node):
        info = self._stack[-1]
        info.ignore = False
        if info.except_line == -1:
            info.except_line = node.lineno
        self._in_except_stack.append(info)
        self.generic_visit(node)
        if hasattr(node, 'end_lineno'):
            info.except_end_line = node.end_lineno
        else:
            info.except_end_line = self.max_line
        self._in_except_stack.pop()
    if sys.version_info[0] >= 3:

        def visit_Raise(self, node):
            for info in self._in_except_stack:
                if node.exc is None:
                    info.raise_lines_in_except.append(node.lineno)
            self.generic_visit(node)
    else:

        def visit_Raise(self, node):
            for info in self._in_except_stack:
                if node.type is None and node.tback is None:
                    info.raise_lines_in_except.append(node.lineno)
            self.generic_visit(node)