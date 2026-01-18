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
def _seq_pprinter_factory(start, end):
    """
    Factory that returns a pprint function useful for sequences.  Used by
    the default pprint for tuples and lists.
    """

    def inner(obj, p, cycle):
        if cycle:
            return p.text(start + '...' + end)
        step = len(start)
        p.begin_group(step, start)
        for idx, x in p._enumerate(obj):
            if idx:
                p.text(',')
                p.breakable()
            p.pretty(x)
        if len(obj) == 1 and isinstance(obj, tuple):
            p.text(',')
        p.end_group(step, end)
    return inner