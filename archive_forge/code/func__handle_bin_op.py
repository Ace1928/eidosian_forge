import ast
import sys
import tokenize
import warnings
from .formatter import (CppFormatter, format_for_loop, format_literal,
from .nodedump import debug_format_node
from .qt import ClassFlag, qt_class_flags
def _handle_bin_op(self, node, op):
    """Handle a binary operator which can appear as 'Augmented Assign'."""
    self.generic_visit(node)
    full_op = f' {op}= ' if self._is_augmented_assign() else f' {op} '
    self._output_file.write(full_op)