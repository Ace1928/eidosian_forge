from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
def _dict2sarray(sorts, ctx):
    sz = len(sorts)
    _names = (Symbol * sz)()
    _sorts = (Sort * sz)()
    i = 0
    for k in sorts:
        v = sorts[k]
        if z3_debug():
            _z3_assert(isinstance(k, str), 'String expected')
            _z3_assert(is_sort(v), 'Z3 sort expected')
        _names[i] = to_symbol(k, ctx)
        _sorts[i] = v.ast
        i = i + 1
    return (sz, _names, _sorts)