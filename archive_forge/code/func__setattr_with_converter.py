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
def _setattr_with_converter(attr_name, value_var, has_on_setattr):
    """
    Use the cached object.setattr to set *attr_name* to *value_var*, but run
    its converter first.
    """
    return "_setattr('%s', %s(%s))" % (attr_name, _init_converter_pat % (attr_name,), value_var)