import __future__
import builtins
import ast
import collections
import contextlib
import doctest
import functools
import os
import re
import string
import sys
import warnings
from pyflakes import messages
class ExportBinding(Binding):
    """
    A binding created by an C{__all__} assignment.  If the names in the list
    can be determined statically, they will be treated as names for export and
    additional checking applied to them.

    The only recognized C{__all__} assignment via list/tuple concatenation is in the
    following format:

        __all__ = ['a'] + ['b'] + ['c']

    Names which are imported and not otherwise used but appear in the value of
    C{__all__} will not have an unused import warning reported for them.
    """

    def __init__(self, name, source, scope):
        if '__all__' in scope and isinstance(source, ast.AugAssign):
            self.names = list(scope['__all__'].names)
        else:
            self.names = []

        def _add_to_names(container):
            for node in container.elts:
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    self.names.append(node.value)
        if isinstance(source.value, (ast.List, ast.Tuple)):
            _add_to_names(source.value)
        elif isinstance(source.value, ast.BinOp):
            currentValue = source.value
            while isinstance(currentValue.right, (ast.List, ast.Tuple)):
                left = currentValue.left
                right = currentValue.right
                _add_to_names(right)
                if isinstance(left, ast.BinOp):
                    currentValue = left
                elif isinstance(left, (ast.List, ast.Tuple)):
                    _add_to_names(left)
                    break
                else:
                    break
        super().__init__(name, source)