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
def _counter_pprint(obj, p, cycle):
    cls_ctor = CallExpression.factory(obj.__class__.__name__)
    if cycle:
        p.pretty(cls_ctor(RawText('...')))
    elif len(obj):
        p.pretty(cls_ctor(dict(obj.most_common())))
    else:
        p.pretty(cls_ctor())