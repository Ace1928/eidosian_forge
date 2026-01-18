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
def _is_name_or_attr(node, name):
    return isinstance(node, ast.Name) and node.id == name or (isinstance(node, ast.Attribute) and node.attr == name)