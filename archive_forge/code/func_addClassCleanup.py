import sys
import functools
import difflib
import pprint
import re
import warnings
import collections
import contextlib
import traceback
import types
from . import result
from .util import (strclass, safe_repr, _count_diff_all_purpose,
@classmethod
def addClassCleanup(cls, function, /, *args, **kwargs):
    """Same as addCleanup, except the cleanup items are called even if
        setUpClass fails (unlike tearDownClass)."""
    cls._class_cleanups.append((function, args, kwargs))