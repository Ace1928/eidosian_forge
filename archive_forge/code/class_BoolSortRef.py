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
class BoolSortRef(SortRef):
    """Boolean sort."""

    def cast(self, val):
        """Try to cast `val` as a Boolean.

        >>> x = BoolSort().cast(True)
        >>> x
        True
        >>> is_expr(x)
        True
        >>> is_expr(True)
        False
        >>> x.sort()
        Bool
        """
        if isinstance(val, bool):
            return BoolVal(val, self.ctx)
        if z3_debug():
            if not is_expr(val):
                msg = 'True, False or Z3 Boolean expression expected. Received %s of type %s'
                _z3_assert(is_expr(val), msg % (val, type(val)))
            if not self.eq(val.sort()):
                _z3_assert(self.eq(val.sort()), 'Value cannot be converted into a Z3 Boolean value')
        return val

    def subsort(self, other):
        return isinstance(other, ArithSortRef)

    def is_int(self):
        return True

    def is_bool(self):
        return True