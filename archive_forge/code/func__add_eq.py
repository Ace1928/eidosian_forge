import contextlib
import copy
import enum
import functools
import inspect
import itertools
import linecache
import sys
import types
import typing
from operator import itemgetter
from . import _compat, _config, setters
from ._compat import (
from .exceptions import (
def _add_eq(cls, attrs=None):
    """
    Add equality methods to *cls* with *attrs*.
    """
    if attrs is None:
        attrs = cls.__attrs_attrs__
    cls.__eq__ = _make_eq(cls, attrs)
    cls.__ne__ = _make_ne()
    return cls