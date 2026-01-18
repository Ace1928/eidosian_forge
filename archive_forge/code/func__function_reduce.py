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
def _function_reduce(self, obj):
    """Reducer for function objects.

        If obj is a top-level attribute of a file-backed module, this
        reducer returns NotImplemented, making the CloudPickler fallback to
        traditional _pickle.Pickler routines to save obj. Otherwise, it reduces
        obj using a custom cloudpickle reducer designed specifically to handle
        dynamic functions.

        As opposed to cloudpickle.py, There no special handling for builtin
        pypy functions because cloudpickle_fast is CPython-specific.
        """
    if _should_pickle_by_reference(obj):
        return NotImplemented
    else:
        return self._dynamic_function_reduce(obj)