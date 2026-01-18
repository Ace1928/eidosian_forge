import builtins
import dis
import opcode
import platform
import sys
import types
import weakref
import uuid
import threading
import typing
import warnings
from .compat import pickle
from collections import OrderedDict
from typing import ClassVar, Generic, Union, Tuple, Callable
from pickle import _getattribute
def _make_skel_func(code, cell_count, base_globals=None):
    """Creates a skeleton function object that contains just the provided
    code and the correct number of cells in func_closure.  All other
    func attributes (e.g. func_globals) are empty.
    """
    warnings.warn('A pickle file created using an old (<=1.4.1) version of cloudpickle is currently being loaded. This is not supported by cloudpickle and will break in cloudpickle 1.7', category=UserWarning)
    if base_globals is None or isinstance(base_globals, str):
        base_globals = {}
    base_globals['__builtins__'] = __builtins__
    closure = tuple((_make_empty_cell() for _ in range(cell_count))) if cell_count >= 0 else None
    return types.FunctionType(code, base_globals, None, None, closure)