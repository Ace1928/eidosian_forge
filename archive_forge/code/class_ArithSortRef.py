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
class ArithSortRef(SortRef):
    """Real and Integer sorts."""

    def is_real(self):
        """Return `True` if `self` is of the sort Real.

        >>> x = Real('x')
        >>> x.is_real()
        True
        >>> (x + 1).is_real()
        True
        >>> x = Int('x')
        >>> x.is_real()
        False
        """
        return self.kind() == Z3_REAL_SORT

    def is_int(self):
        """Return `True` if `self` is of the sort Integer.

        >>> x = Int('x')
        >>> x.is_int()
        True
        >>> (x + 1).is_int()
        True
        >>> x = Real('x')
        >>> x.is_int()
        False
        """
        return self.kind() == Z3_INT_SORT

    def is_bool(self):
        return False

    def subsort(self, other):
        """Return `True` if `self` is a subsort of `other`."""
        return self.is_int() and is_arith_sort(other) and other.is_real()

    def cast(self, val):
        """Try to cast `val` as an Integer or Real.

        >>> IntSort().cast(10)
        10
        >>> is_int(IntSort().cast(10))
        True
        >>> is_int(10)
        False
        >>> RealSort().cast(10)
        10
        >>> is_real(RealSort().cast(10))
        True
        """
        if is_expr(val):
            if z3_debug():
                _z3_assert(self.ctx == val.ctx, 'Context mismatch')
            val_s = val.sort()
            if self.eq(val_s):
                return val
            if val_s.is_int() and self.is_real():
                return ToReal(val)
            if val_s.is_bool() and self.is_int():
                return If(val, 1, 0)
            if val_s.is_bool() and self.is_real():
                return ToReal(If(val, 1, 0))
            if z3_debug():
                _z3_assert(False, 'Z3 Integer/Real expression expected')
        else:
            if self.is_int():
                return IntVal(val, self.ctx)
            if self.is_real():
                return RealVal(val, self.ctx)
            if z3_debug():
                msg = 'int, long, float, string (numeral), or Z3 Integer/Real expression expected. Got %s'
                _z3_assert(False, msg % self)