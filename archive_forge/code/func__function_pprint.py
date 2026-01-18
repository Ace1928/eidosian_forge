from contextlib import contextmanager
import datetime
import os
import re
import sys
import types
from collections import deque
from inspect import signature
from io import StringIO
from warnings import warn
from IPython.utils.decorators import undoc
from IPython.utils.py3compat import PYPY
from typing import Dict
def _function_pprint(obj, p, cycle):
    """Base pprint for all functions and builtin functions."""
    name = _safe_getattr(obj, '__qualname__', obj.__name__)
    mod = obj.__module__
    if mod and mod not in ('__builtin__', 'builtins', 'exceptions'):
        name = mod + '.' + name
    try:
        func_def = name + str(signature(obj))
    except ValueError:
        func_def = name
    p.text('<function %s>' % func_def)