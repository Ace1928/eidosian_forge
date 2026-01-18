import ast
import sys
import tokenize
import warnings
from .formatter import (CppFormatter, format_for_loop, format_literal,
from .nodedump import debug_format_node
from .qt import ClassFlag, qt_class_flags
def _ignore_function_def_node(self, node):
    """Should this node be ignored within a FunctionDef."""
    if not self._stack:
        return False
    parent = self._stack[-1]
    if isinstance(parent, (ast.arguments, ast.arg)):
        return True
    if not isinstance(parent, ast.FunctionDef):
        return False
    return node == parent.returns or node in parent.decorator_list