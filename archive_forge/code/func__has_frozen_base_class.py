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
def _has_frozen_base_class(cls):
    """
    Check whether *cls* has a frozen ancestor by looking at its
    __setattr__.
    """
    return cls.__setattr__ is _frozen_setattrs