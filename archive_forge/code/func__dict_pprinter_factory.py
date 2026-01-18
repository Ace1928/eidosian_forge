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
def _dict_pprinter_factory(start, end):
    """
    Factory that returns a pprint function used by the default pprint of
    dicts and dict proxies.
    """

    def inner(obj, p, cycle):
        if cycle:
            return p.text('{...}')
        step = len(start)
        p.begin_group(step, start)
        keys = obj.keys()
        for idx, key in p._enumerate(keys):
            if idx:
                p.text(',')
                p.breakable()
            p.pretty(key)
            p.text(': ')
            p.pretty(obj[key])
        p.end_group(step, end)
    return inner