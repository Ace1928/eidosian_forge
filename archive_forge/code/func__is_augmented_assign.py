import ast
import sys
import tokenize
import warnings
from .formatter import (CppFormatter, format_for_loop, format_literal,
from .nodedump import debug_format_node
from .qt import ClassFlag, qt_class_flags
def _is_augmented_assign(self):
    """Is it 'Augmented_assign' (operators +=/-=, etc)?"""
    return self._stack and isinstance(self._stack[-1], ast.AugAssign)