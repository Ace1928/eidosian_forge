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
def COMPARE(self, node):
    left = node.left
    for op, right in zip(node.ops, node.comparators):
        if isinstance(op, (ast.Is, ast.IsNot)) and (_is_const_non_singleton(left) or _is_const_non_singleton(right)):
            self.report(messages.IsLiteral, node)
        left = right
    self.handleChildren(node)