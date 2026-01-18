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
def DeclareSort(name, ctx=None):
    """Create a new uninterpreted sort named `name`.

    If `ctx=None`, then the new sort is declared in the global Z3Py context.

    >>> A = DeclareSort('A')
    >>> a = Const('a', A)
    >>> b = Const('b', A)
    >>> a.sort() == A
    True
    >>> b.sort() == A
    True
    >>> a == b
    a == b
    """
    ctx = _get_ctx(ctx)
    return SortRef(Z3_mk_uninterpreted_sort(ctx.ref(), to_symbol(name, ctx)), ctx)