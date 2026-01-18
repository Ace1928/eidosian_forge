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
def _add_hash(cls, attrs):
    """
    Add a hash method to *cls*.
    """
    cls.__hash__ = _make_hash(cls, attrs, frozen=False, cache_hash=False)
    return cls