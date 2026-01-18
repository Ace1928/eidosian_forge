import ast
import sys
import tokenize
import warnings
from .formatter import (CppFormatter, format_for_loop, format_literal,
from .nodedump import debug_format_node
from .qt import ClassFlag, qt_class_flags
def _is_if_main(if_node):
    """Return whether an if statement is: if __name__ == '__main__' """
    test = if_node.test
    return isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq) and isinstance(test.left, ast.Name) and (test.left.id == '__name__') and (len(test.comparators) == 1) and isinstance(test.comparators[0], ast.Constant) and (test.comparators[0].value == '__main__')