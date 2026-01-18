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
def _add_repr(cls, ns=None, attrs=None):
    """
    Add a repr method to *cls*.
    """
    if attrs is None:
        attrs = cls.__attrs_attrs__
    cls.__repr__ = _make_repr(attrs, ns, cls)
    return cls