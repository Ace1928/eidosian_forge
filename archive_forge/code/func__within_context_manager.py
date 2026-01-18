import ast
import sys
import tokenize
import warnings
from .formatter import (CppFormatter, format_for_loop, format_literal,
from .nodedump import debug_format_node
from .qt import ClassFlag, qt_class_flags
def _within_context_manager(self):
    """Return whether we are within a context manager (with)."""
    parent = self._stack[-1] if self._stack else None
    return parent and isinstance(parent, ast.withitem)