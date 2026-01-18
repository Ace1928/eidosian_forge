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
def _coerce_fp_expr_list(alist, ctx):
    first_fp_sort = None
    for a in alist:
        if is_fp(a):
            if first_fp_sort is None:
                first_fp_sort = a.sort()
            elif first_fp_sort == a.sort():
                pass
            else:
                first_fp_sort = None
                break
    r = []
    for i in range(len(alist)):
        a = alist[i]
        is_repr = isinstance(a, str) and a.contains('2**(') and a.endswith(')')
        if is_repr or _is_int(a) or isinstance(a, (float, bool)):
            r.append(FPVal(a, None, first_fp_sort, ctx))
        else:
            r.append(a)
    return _coerce_expr_list(r, ctx)