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
def _is_const_non_singleton(node):
    return _is_constant(node) and (not _is_singleton(node))