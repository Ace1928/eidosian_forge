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
def _module_reduce(obj):
    if _should_pickle_by_reference(obj):
        return (subimport, (obj.__name__,))
    else:
        state = obj.__dict__.copy()
        state.pop('__builtins__', None)
        return (dynamic_subimport, (obj.__name__, state))