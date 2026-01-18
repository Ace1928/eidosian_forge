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
def _type_pprint(obj, p, cycle):
    """The pprint for classes and types."""
    if [m for m in _get_mro(type(obj)) if '__repr__' in vars(m)][:1] != [type]:
        _repr_pprint(obj, p, cycle)
        return
    mod = _safe_getattr(obj, '__module__', None)
    try:
        name = obj.__qualname__
        if not isinstance(name, str):
            raise Exception('Try __name__')
    except Exception:
        name = obj.__name__
        if not isinstance(name, str):
            name = '<unknown type>'
    if mod in (None, '__builtin__', 'builtins', 'exceptions'):
        p.text(name)
    else:
        p.text(mod + '.' + name)