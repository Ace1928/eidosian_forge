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
def _super_pprint(obj, p, cycle):
    """The pprint for the super type."""
    p.begin_group(8, '<super: ')
    p.pretty(obj.__thisclass__)
    p.text(',')
    p.breakable()
    if PYPY:
        dself = obj.__repr__.__self__
        p.pretty(None if dself is obj else dself)
    else:
        p.pretty(obj.__self__)
    p.end_group(8, '>')