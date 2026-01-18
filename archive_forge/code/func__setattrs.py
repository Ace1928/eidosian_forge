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
def _setattrs(self, name_values_pairs):
    bound_setattr = _obj_setattr.__get__(self)
    for name, value in name_values_pairs:
        if name != 'metadata':
            bound_setattr(name, value)
        else:
            bound_setattr(name, types.MappingProxyType(dict(value)) if value else _empty_metadata_singleton)