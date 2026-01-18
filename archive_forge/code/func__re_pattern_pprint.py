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
def _re_pattern_pprint(obj, p, cycle):
    """The pprint function for regular expression patterns."""
    re_compile = CallExpression.factory('re.compile')
    if obj.flags:
        p.pretty(re_compile(RawStringLiteral(obj.pattern), _ReFlags(obj.flags)))
    else:
        p.pretty(re_compile(RawStringLiteral(obj.pattern)))