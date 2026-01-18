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
def _collect_base_attrs_broken(cls, taken_attr_names):
    """
    Collect attr.ibs from base classes of *cls*, except *taken_attr_names*.

    N.B. *taken_attr_names* will be mutated.

    Adhere to the old incorrect behavior.

    Notably it collects from the front and considers inherited attributes which
    leads to the buggy behavior reported in #428.
    """
    base_attrs = []
    base_attr_map = {}
    for base_cls in cls.__mro__[1:-1]:
        for a in getattr(base_cls, '__attrs_attrs__', []):
            if a.name in taken_attr_names:
                continue
            a = a.evolve(inherited=True)
            taken_attr_names.add(a.name)
            base_attrs.append(a)
            base_attr_map[a.name] = base_cls
    return (base_attrs, base_attr_map)