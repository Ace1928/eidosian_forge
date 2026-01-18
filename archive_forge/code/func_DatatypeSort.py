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
def DatatypeSort(name, ctx=None):
    """Create a reference to a sort that was declared, or will be declared, as a recursive datatype"""
    ctx = _get_ctx(ctx)
    return DatatypeSortRef(Z3_mk_datatype_sort(ctx.ref(), to_symbol(name, ctx)), ctx)