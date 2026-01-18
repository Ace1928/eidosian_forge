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
def _set_pprinter_factory(start, end):
    """
    Factory that returns a pprint function useful for sets and frozensets.
    """

    def inner(obj, p, cycle):
        if cycle:
            return p.text(start + '...' + end)
        if len(obj) == 0:
            p.text(type(obj).__name__ + '()')
        else:
            step = len(start)
            p.begin_group(step, start)
            if not (p.max_seq_length and len(obj) >= p.max_seq_length):
                items = _sorted_for_pprint(obj)
            else:
                items = obj
            for idx, x in p._enumerate(items):
                if idx:
                    p.text(',')
                    p.breakable()
                p.pretty(x)
            p.end_group(step, end)
    return inner