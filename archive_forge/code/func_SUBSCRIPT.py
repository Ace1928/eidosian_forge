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
def SUBSCRIPT(self, node):
    if _is_name_or_attr(node.value, 'Literal'):
        with self._enter_annotation(AnnotationState.NONE):
            self.handleChildren(node)
    elif _is_name_or_attr(node.value, 'Annotated'):
        self.handleNode(node.value, node)
        if isinstance(node.slice, ast.Tuple):
            slice_tuple = node.slice
        elif isinstance(node.slice, ast.Index) and isinstance(node.slice.value, ast.Tuple):
            slice_tuple = node.slice.value
        else:
            slice_tuple = None
        if slice_tuple is None or len(slice_tuple.elts) < 2:
            self.handleNode(node.slice, node)
        else:
            self.handleNode(slice_tuple.elts[0], node)
            with self._enter_annotation(AnnotationState.NONE):
                for arg in slice_tuple.elts[1:]:
                    self.handleNode(arg, node)
        self.handleNode(node.ctx, node)
    elif _is_any_typing_member(node.value, self.scopeStack):
        with self._enter_annotation():
            self.handleChildren(node)
    else:
        self.handleChildren(node)