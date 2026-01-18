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
def add_eq(self):
    cd = self._cls_dict
    cd['__eq__'] = self._add_method_dunders(_make_eq(self._cls, self._attrs))
    cd['__ne__'] = self._add_method_dunders(_make_ne())
    return self