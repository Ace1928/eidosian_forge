import ast
import sys
import tokenize
import warnings
from .formatter import (CppFormatter, format_for_loop, format_literal,
from .nodedump import debug_format_node
from .qt import ClassFlag, qt_class_flags
def _write_function_args(self, args_node):
    for i, arg in enumerate(args_node):
        if i > 0:
            self._output_file.write(', ')
        self.visit(arg)