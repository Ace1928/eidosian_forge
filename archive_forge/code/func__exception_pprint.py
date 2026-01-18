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
def _exception_pprint(obj, p, cycle):
    """Base pprint for all exceptions."""
    name = getattr(obj.__class__, '__qualname__', obj.__class__.__name__)
    if obj.__class__.__module__ not in ('exceptions', 'builtins'):
        name = '%s.%s' % (obj.__class__.__module__, name)
    p.pretty(CallExpression(name, *getattr(obj, 'args', ())))