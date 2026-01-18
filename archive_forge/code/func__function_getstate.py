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
def _function_getstate(func):
    slotstate = {'__name__': func.__name__, '__qualname__': func.__qualname__, '__annotations__': func.__annotations__, '__kwdefaults__': func.__kwdefaults__, '__defaults__': func.__defaults__, '__module__': func.__module__, '__doc__': func.__doc__, '__closure__': func.__closure__}
    f_globals_ref = _extract_code_globals(func.__code__)
    f_globals = {k: func.__globals__[k] for k in f_globals_ref if k in func.__globals__}
    closure_values = list(map(_get_cell_contents, func.__closure__)) if func.__closure__ is not None else ()
    slotstate['_cloudpickle_submodules'] = _find_imported_submodules(func.__code__, itertools.chain(f_globals.values(), closure_values))
    slotstate['__globals__'] = f_globals
    state = func.__dict__
    return (state, slotstate)