import _collections_abc
import abc
import copyreg
import io
import itertools
import logging
import sys
import struct
import types
import weakref
import typing
from enum import Enum
from collections import ChainMap, OrderedDict
from .compat import pickle, Pickler
from .cloudpickle import (
def _dynamic_function_reduce(self, func):
    """Reduce a function that is not pickleable via attribute lookup."""
    newargs = self._function_getnewargs(func)
    state = _function_getstate(func)
    return (_make_function, newargs, state, None, None, _function_setstate)