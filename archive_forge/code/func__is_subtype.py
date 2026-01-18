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
def _is_subtype(expected, basetype):
    if isinstance(expected, tuple):
        return all((_is_subtype(e, basetype) for e in expected))
    return isinstance(expected, type) and issubclass(expected, basetype)