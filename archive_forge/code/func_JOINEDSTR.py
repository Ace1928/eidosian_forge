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
def JOINEDSTR(self, node):
    if not self._in_fstring and (not any((isinstance(x, ast.FormattedValue) for x in node.values))):
        self.report(messages.FStringMissingPlaceholders, node)
    self._in_fstring, orig = (True, self._in_fstring)
    try:
        self.handleChildren(node)
    finally:
        self._in_fstring = orig