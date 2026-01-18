import math
import ctypes
import copy
from .random import uniform
from .symbol import Symbol
from . import symbol
from ..base import _LIB, check_call
from ..base import SymbolHandle, _as_list
from ..attribute import AttrScope
def _cond_wrapper(loop_vars):
    result = cond(*loop_vars)
    if not isinstance(result, Symbol):
        raise ValueError('Return of cond must be a Symbol')
    return ([], [result], [], [])