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
class FPRef(ExprRef):
    """Floating-point expressions."""

    def sort(self):
        """Return the sort of the floating-point expression `self`.

        >>> x = FP('1.0', FPSort(8, 24))
        >>> x.sort()
        FPSort(8, 24)
        >>> x.sort() == FPSort(8, 24)
        True
        """
        return FPSortRef(Z3_get_sort(self.ctx_ref(), self.as_ast()), self.ctx)

    def ebits(self):
        """Retrieves the number of bits reserved for the exponent in the FloatingPoint expression `self`.
        >>> b = FPSort(8, 24)
        >>> b.ebits()
        8
        """
        return self.sort().ebits()

    def sbits(self):
        """Retrieves the number of bits reserved for the exponent in the FloatingPoint expression `self`.
        >>> b = FPSort(8, 24)
        >>> b.sbits()
        24
        """
        return self.sort().sbits()

    def as_string(self):
        """Return a Z3 floating point expression as a Python string."""
        return Z3_ast_to_string(self.ctx_ref(), self.as_ast())

    def __le__(self, other):
        return fpLEQ(self, other, self.ctx)

    def __lt__(self, other):
        return fpLT(self, other, self.ctx)

    def __ge__(self, other):
        return fpGEQ(self, other, self.ctx)

    def __gt__(self, other):
        return fpGT(self, other, self.ctx)

    def __add__(self, other):
        """Create the Z3 expression `self + other`.

        >>> x = FP('x', FPSort(8, 24))
        >>> y = FP('y', FPSort(8, 24))
        >>> x + y
        x + y
        >>> (x + y).sort()
        FPSort(8, 24)
        """
        [a, b] = _coerce_fp_expr_list([self, other], self.ctx)
        return fpAdd(_dflt_rm(), a, b, self.ctx)

    def __radd__(self, other):
        """Create the Z3 expression `other + self`.

        >>> x = FP('x', FPSort(8, 24))
        >>> 10 + x
        1.25*(2**3) + x
        """
        [a, b] = _coerce_fp_expr_list([other, self], self.ctx)
        return fpAdd(_dflt_rm(), a, b, self.ctx)

    def __sub__(self, other):
        """Create the Z3 expression `self - other`.

        >>> x = FP('x', FPSort(8, 24))
        >>> y = FP('y', FPSort(8, 24))
        >>> x - y
        x - y
        >>> (x - y).sort()
        FPSort(8, 24)
        """
        [a, b] = _coerce_fp_expr_list([self, other], self.ctx)
        return fpSub(_dflt_rm(), a, b, self.ctx)

    def __rsub__(self, other):
        """Create the Z3 expression `other - self`.

        >>> x = FP('x', FPSort(8, 24))
        >>> 10 - x
        1.25*(2**3) - x
        """
        [a, b] = _coerce_fp_expr_list([other, self], self.ctx)
        return fpSub(_dflt_rm(), a, b, self.ctx)

    def __mul__(self, other):
        """Create the Z3 expression `self * other`.

        >>> x = FP('x', FPSort(8, 24))
        >>> y = FP('y', FPSort(8, 24))
        >>> x * y
        x * y
        >>> (x * y).sort()
        FPSort(8, 24)
        >>> 10 * y
        1.25*(2**3) * y
        """
        [a, b] = _coerce_fp_expr_list([self, other], self.ctx)
        return fpMul(_dflt_rm(), a, b, self.ctx)

    def __rmul__(self, other):
        """Create the Z3 expression `other * self`.

        >>> x = FP('x', FPSort(8, 24))
        >>> y = FP('y', FPSort(8, 24))
        >>> x * y
        x * y
        >>> x * 10
        x * 1.25*(2**3)
        """
        [a, b] = _coerce_fp_expr_list([other, self], self.ctx)
        return fpMul(_dflt_rm(), a, b, self.ctx)

    def __pos__(self):
        """Create the Z3 expression `+self`."""
        return self

    def __neg__(self):
        """Create the Z3 expression `-self`.

        >>> x = FP('x', Float32())
        >>> -x
        -x
        """
        return fpNeg(self)

    def __div__(self, other):
        """Create the Z3 expression `self / other`.

        >>> x = FP('x', FPSort(8, 24))
        >>> y = FP('y', FPSort(8, 24))
        >>> x / y
        x / y
        >>> (x / y).sort()
        FPSort(8, 24)
        >>> 10 / y
        1.25*(2**3) / y
        """
        [a, b] = _coerce_fp_expr_list([self, other], self.ctx)
        return fpDiv(_dflt_rm(), a, b, self.ctx)

    def __rdiv__(self, other):
        """Create the Z3 expression `other / self`.

        >>> x = FP('x', FPSort(8, 24))
        >>> y = FP('y', FPSort(8, 24))
        >>> x / y
        x / y
        >>> x / 10
        x / 1.25*(2**3)
        """
        [a, b] = _coerce_fp_expr_list([other, self], self.ctx)
        return fpDiv(_dflt_rm(), a, b, self.ctx)

    def __truediv__(self, other):
        """Create the Z3 expression division `self / other`."""
        return self.__div__(other)

    def __rtruediv__(self, other):
        """Create the Z3 expression division `other / self`."""
        return self.__rdiv__(other)

    def __mod__(self, other):
        """Create the Z3 expression mod `self % other`."""
        return fpRem(self, other)

    def __rmod__(self, other):
        """Create the Z3 expression mod `other % self`."""
        return fpRem(other, self)