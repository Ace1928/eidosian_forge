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
def _sorted_for_pprint(items):
    """
    Sort the given items for pretty printing. Since some predictable
    sorting is better than no sorting at all, we sort on the string
    representation if normal sorting fails.
    """
    items = list(items)
    try:
        return sorted(items)
    except Exception:
        try:
            return sorted(items, key=str)
        except Exception:
            return items