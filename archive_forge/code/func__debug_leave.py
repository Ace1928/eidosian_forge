import ast
import sys
import tokenize
import warnings
from .formatter import (CppFormatter, format_for_loop, format_literal,
from .nodedump import debug_format_node
from .qt import ClassFlag, qt_class_flags
def _debug_leave(self, node):
    self._debug_indent -= 1
    message = '{}<generic_visit({})\n'.format('  ' * self._debug_indent, type(node).__name__)
    sys.stderr.write(message)