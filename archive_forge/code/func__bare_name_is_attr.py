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
def _bare_name_is_attr(name):
    for scope in reversed(scope_stack):
        if name in scope:
            return isinstance(scope[name], ImportationFrom) and scope[name].module in TYPING_MODULES and is_name_match_fn(scope[name].real_name)
    return False