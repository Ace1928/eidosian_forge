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
def add_hash(self):
    self._cls_dict['__hash__'] = self._add_method_dunders(_make_hash(self._cls, self._attrs, frozen=self._frozen, cache_hash=self._cache_hash))
    return self