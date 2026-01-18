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
def CharSort(ctx=None):
    """Create a character sort
    >>> ch = CharSort()
    >>> print(ch)
    Char
    """
    ctx = _get_ctx(ctx)
    return CharSortRef(Z3_mk_char_sort(ctx.ref()), ctx)