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
def _is_tuple_constant(node):
    return isinstance(node, ast.Tuple) and all((_is_constant(elt) for elt in node.elts))