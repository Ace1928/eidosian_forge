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
def _get_annotations(cls):
    """
    Get annotations for *cls*.
    """
    if _has_own_attribute(cls, '__annotations__'):
        return cls.__annotations__
    return {}