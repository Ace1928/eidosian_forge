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
class _ReFlags:

    def __init__(self, value):
        self.value = value

    def _repr_pretty_(self, p, cycle):
        done_one = False
        for flag in ('TEMPLATE', 'IGNORECASE', 'LOCALE', 'MULTILINE', 'DOTALL', 'UNICODE', 'VERBOSE', 'DEBUG'):
            if self.value & getattr(re, flag):
                if done_one:
                    p.text('|')
                p.text('re.' + flag)
                done_one = True