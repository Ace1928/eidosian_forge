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
def _collect_base_attrs(cls, taken_attr_names):
    """
    Collect attr.ibs from base classes of *cls*, except *taken_attr_names*.
    """
    base_attrs = []
    base_attr_map = {}
    for base_cls in reversed(cls.__mro__[1:-1]):
        for a in getattr(base_cls, '__attrs_attrs__', []):
            if a.inherited or a.name in taken_attr_names:
                continue
            a = a.evolve(inherited=True)
            base_attrs.append(a)
            base_attr_map[a.name] = base_cls
    filtered = []
    seen = set()
    for a in reversed(base_attrs):
        if a.name in seen:
            continue
        filtered.insert(0, a)
        seen.add(a.name)
    return (filtered, base_attr_map)