import _collections_abc
import abc
import copyreg
import io
import itertools
import logging
import sys
import struct
import types
import weakref
import typing
from enum import Enum
from collections import ChainMap, OrderedDict
from .compat import pickle, Pickler
from .cloudpickle import (
def _function_getnewargs(self, func):
    code = func.__code__
    base_globals = self.globals_ref.setdefault(id(func.__globals__), {})
    if base_globals == {}:
        for k in ['__package__', '__name__', '__path__', '__file__']:
            if k in func.__globals__:
                base_globals[k] = func.__globals__[k]
    if func.__closure__ is None:
        closure = None
    else:
        closure = tuple((_make_empty_cell() for _ in range(len(code.co_freevars))))
    return (code, base_globals, None, None, closure)