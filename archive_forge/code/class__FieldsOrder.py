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
class _FieldsOrder(dict):
    """Fix order of AST node fields."""

    def _get_fields(self, node_class):
        fields = node_class._fields
        if 'iter' in fields:
            key_first = 'iter'.find
        elif 'generators' in fields:
            key_first = 'generators'.find
        else:
            key_first = 'value'.find
        return tuple(sorted(fields, key=key_first, reverse=True))

    def __missing__(self, node_class):
        self[node_class] = fields = self._get_fields(node_class)
        return fields