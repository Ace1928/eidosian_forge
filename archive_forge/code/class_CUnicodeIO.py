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
@undoc
class CUnicodeIO(StringIO):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warn('CUnicodeIO is deprecated since IPython 6.0. Please use io.StringIO instead.', DeprecationWarning, stacklevel=2)