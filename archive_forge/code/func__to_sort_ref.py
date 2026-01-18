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
def _to_sort_ref(s, ctx):
    if z3_debug():
        _z3_assert(isinstance(s, Sort), 'Z3 Sort expected')
    k = _sort_kind(ctx, s)
    if k == Z3_BOOL_SORT:
        return BoolSortRef(s, ctx)
    elif k == Z3_INT_SORT or k == Z3_REAL_SORT:
        return ArithSortRef(s, ctx)
    elif k == Z3_BV_SORT:
        return BitVecSortRef(s, ctx)
    elif k == Z3_ARRAY_SORT:
        return ArraySortRef(s, ctx)
    elif k == Z3_DATATYPE_SORT:
        return DatatypeSortRef(s, ctx)
    elif k == Z3_FINITE_DOMAIN_SORT:
        return FiniteDomainSortRef(s, ctx)
    elif k == Z3_FLOATING_POINT_SORT:
        return FPSortRef(s, ctx)
    elif k == Z3_ROUNDING_MODE_SORT:
        return FPRMSortRef(s, ctx)
    elif k == Z3_RE_SORT:
        return ReSortRef(s, ctx)
    elif k == Z3_SEQ_SORT:
        return SeqSortRef(s, ctx)
    elif k == Z3_CHAR_SORT:
        return CharSortRef(s, ctx)
    elif k == Z3_TYPE_VAR:
        return TypeVarRef(s, ctx)
    return SortRef(s, ctx)