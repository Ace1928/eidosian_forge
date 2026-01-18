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
def _repr_pprint(obj, p, cycle):
    """A pprint that just redirects to the normal repr function."""
    output = repr(obj)
    lines = output.splitlines()
    with p.group():
        for idx, output_line in enumerate(lines):
            if idx:
                p.break_()
            p.text(output_line)