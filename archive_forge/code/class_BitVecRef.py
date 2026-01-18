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
class BitVecRef(ExprRef):
    """Bit-vector expressions."""

    def sort(self):
        """Return the sort of the bit-vector expression `self`.

        >>> x = BitVec('x', 32)
        >>> x.sort()
        BitVec(32)
        >>> x.sort() == BitVecSort(32)
        True
        """
        return BitVecSortRef(Z3_get_sort(self.ctx_ref(), self.as_ast()), self.ctx)

    def size(self):
        """Return the number of bits of the bit-vector expression `self`.

        >>> x = BitVec('x', 32)
        >>> (x + 1).size()
        32
        >>> Concat(x, x).size()
        64
        """
        return self.sort().size()

    def __add__(self, other):
        """Create the Z3 expression `self + other`.

        >>> x = BitVec('x', 32)
        >>> y = BitVec('y', 32)
        >>> x + y
        x + y
        >>> (x + y).sort()
        BitVec(32)
        """
        a, b = _coerce_exprs(self, other)
        return BitVecRef(Z3_mk_bvadd(self.ctx_ref(), a.as_ast(), b.as_ast()), self.ctx)

    def __radd__(self, other):
        """Create the Z3 expression `other + self`.

        >>> x = BitVec('x', 32)
        >>> 10 + x
        10 + x
        """
        a, b = _coerce_exprs(self, other)
        return BitVecRef(Z3_mk_bvadd(self.ctx_ref(), b.as_ast(), a.as_ast()), self.ctx)

    def __mul__(self, other):
        """Create the Z3 expression `self * other`.

        >>> x = BitVec('x', 32)
        >>> y = BitVec('y', 32)
        >>> x * y
        x*y
        >>> (x * y).sort()
        BitVec(32)
        """
        a, b = _coerce_exprs(self, other)
        return BitVecRef(Z3_mk_bvmul(self.ctx_ref(), a.as_ast(), b.as_ast()), self.ctx)

    def __rmul__(self, other):
        """Create the Z3 expression `other * self`.

        >>> x = BitVec('x', 32)
        >>> 10 * x
        10*x
        """
        a, b = _coerce_exprs(self, other)
        return BitVecRef(Z3_mk_bvmul(self.ctx_ref(), b.as_ast(), a.as_ast()), self.ctx)

    def __sub__(self, other):
        """Create the Z3 expression `self - other`.

        >>> x = BitVec('x', 32)
        >>> y = BitVec('y', 32)
        >>> x - y
        x - y
        >>> (x - y).sort()
        BitVec(32)
        """
        a, b = _coerce_exprs(self, other)
        return BitVecRef(Z3_mk_bvsub(self.ctx_ref(), a.as_ast(), b.as_ast()), self.ctx)

    def __rsub__(self, other):
        """Create the Z3 expression `other - self`.

        >>> x = BitVec('x', 32)
        >>> 10 - x
        10 - x
        """
        a, b = _coerce_exprs(self, other)
        return BitVecRef(Z3_mk_bvsub(self.ctx_ref(), b.as_ast(), a.as_ast()), self.ctx)

    def __or__(self, other):
        """Create the Z3 expression bitwise-or `self | other`.

        >>> x = BitVec('x', 32)
        >>> y = BitVec('y', 32)
        >>> x | y
        x | y
        >>> (x | y).sort()
        BitVec(32)
        """
        a, b = _coerce_exprs(self, other)
        return BitVecRef(Z3_mk_bvor(self.ctx_ref(), a.as_ast(), b.as_ast()), self.ctx)

    def __ror__(self, other):
        """Create the Z3 expression bitwise-or `other | self`.

        >>> x = BitVec('x', 32)
        >>> 10 | x
        10 | x
        """
        a, b = _coerce_exprs(self, other)
        return BitVecRef(Z3_mk_bvor(self.ctx_ref(), b.as_ast(), a.as_ast()), self.ctx)

    def __and__(self, other):
        """Create the Z3 expression bitwise-and `self & other`.

        >>> x = BitVec('x', 32)
        >>> y = BitVec('y', 32)
        >>> x & y
        x & y
        >>> (x & y).sort()
        BitVec(32)
        """
        a, b = _coerce_exprs(self, other)
        return BitVecRef(Z3_mk_bvand(self.ctx_ref(), a.as_ast(), b.as_ast()), self.ctx)

    def __rand__(self, other):
        """Create the Z3 expression bitwise-or `other & self`.

        >>> x = BitVec('x', 32)
        >>> 10 & x
        10 & x
        """
        a, b = _coerce_exprs(self, other)
        return BitVecRef(Z3_mk_bvand(self.ctx_ref(), b.as_ast(), a.as_ast()), self.ctx)

    def __xor__(self, other):
        """Create the Z3 expression bitwise-xor `self ^ other`.

        >>> x = BitVec('x', 32)
        >>> y = BitVec('y', 32)
        >>> x ^ y
        x ^ y
        >>> (x ^ y).sort()
        BitVec(32)
        """
        a, b = _coerce_exprs(self, other)
        return BitVecRef(Z3_mk_bvxor(self.ctx_ref(), a.as_ast(), b.as_ast()), self.ctx)

    def __rxor__(self, other):
        """Create the Z3 expression bitwise-xor `other ^ self`.

        >>> x = BitVec('x', 32)
        >>> 10 ^ x
        10 ^ x
        """
        a, b = _coerce_exprs(self, other)
        return BitVecRef(Z3_mk_bvxor(self.ctx_ref(), b.as_ast(), a.as_ast()), self.ctx)

    def __pos__(self):
        """Return `self`.

        >>> x = BitVec('x', 32)
        >>> +x
        x
        """
        return self

    def __neg__(self):
        """Return an expression representing `-self`.

        >>> x = BitVec('x', 32)
        >>> -x
        -x
        >>> simplify(-(-x))
        x
        """
        return BitVecRef(Z3_mk_bvneg(self.ctx_ref(), self.as_ast()), self.ctx)

    def __invert__(self):
        """Create the Z3 expression bitwise-not `~self`.

        >>> x = BitVec('x', 32)
        >>> ~x
        ~x
        >>> simplify(~(~x))
        x
        """
        return BitVecRef(Z3_mk_bvnot(self.ctx_ref(), self.as_ast()), self.ctx)

    def __div__(self, other):
        """Create the Z3 expression (signed) division `self / other`.

        Use the function UDiv() for unsigned division.

        >>> x = BitVec('x', 32)
        >>> y = BitVec('y', 32)
        >>> x / y
        x/y
        >>> (x / y).sort()
        BitVec(32)
        >>> (x / y).sexpr()
        '(bvsdiv x y)'
        >>> UDiv(x, y).sexpr()
        '(bvudiv x y)'
        """
        a, b = _coerce_exprs(self, other)
        return BitVecRef(Z3_mk_bvsdiv(self.ctx_ref(), a.as_ast(), b.as_ast()), self.ctx)

    def __truediv__(self, other):
        """Create the Z3 expression (signed) division `self / other`."""
        return self.__div__(other)

    def __rdiv__(self, other):
        """Create the Z3 expression (signed) division `other / self`.

        Use the function UDiv() for unsigned division.

        >>> x = BitVec('x', 32)
        >>> 10 / x
        10/x
        >>> (10 / x).sexpr()
        '(bvsdiv #x0000000a x)'
        >>> UDiv(10, x).sexpr()
        '(bvudiv #x0000000a x)'
        """
        a, b = _coerce_exprs(self, other)
        return BitVecRef(Z3_mk_bvsdiv(self.ctx_ref(), b.as_ast(), a.as_ast()), self.ctx)

    def __rtruediv__(self, other):
        """Create the Z3 expression (signed) division `other / self`."""
        return self.__rdiv__(other)

    def __mod__(self, other):
        """Create the Z3 expression (signed) mod `self % other`.

        Use the function URem() for unsigned remainder, and SRem() for signed remainder.

        >>> x = BitVec('x', 32)
        >>> y = BitVec('y', 32)
        >>> x % y
        x%y
        >>> (x % y).sort()
        BitVec(32)
        >>> (x % y).sexpr()
        '(bvsmod x y)'
        >>> URem(x, y).sexpr()
        '(bvurem x y)'
        >>> SRem(x, y).sexpr()
        '(bvsrem x y)'
        """
        a, b = _coerce_exprs(self, other)
        return BitVecRef(Z3_mk_bvsmod(self.ctx_ref(), a.as_ast(), b.as_ast()), self.ctx)

    def __rmod__(self, other):
        """Create the Z3 expression (signed) mod `other % self`.

        Use the function URem() for unsigned remainder, and SRem() for signed remainder.

        >>> x = BitVec('x', 32)
        >>> 10 % x
        10%x
        >>> (10 % x).sexpr()
        '(bvsmod #x0000000a x)'
        >>> URem(10, x).sexpr()
        '(bvurem #x0000000a x)'
        >>> SRem(10, x).sexpr()
        '(bvsrem #x0000000a x)'
        """
        a, b = _coerce_exprs(self, other)
        return BitVecRef(Z3_mk_bvsmod(self.ctx_ref(), b.as_ast(), a.as_ast()), self.ctx)

    def __le__(self, other):
        """Create the Z3 expression (signed) `other <= self`.

        Use the function ULE() for unsigned less than or equal to.

        >>> x, y = BitVecs('x y', 32)
        >>> x <= y
        x <= y
        >>> (x <= y).sexpr()
        '(bvsle x y)'
        >>> ULE(x, y).sexpr()
        '(bvule x y)'
        """
        a, b = _coerce_exprs(self, other)
        return BoolRef(Z3_mk_bvsle(self.ctx_ref(), a.as_ast(), b.as_ast()), self.ctx)

    def __lt__(self, other):
        """Create the Z3 expression (signed) `other < self`.

        Use the function ULT() for unsigned less than.

        >>> x, y = BitVecs('x y', 32)
        >>> x < y
        x < y
        >>> (x < y).sexpr()
        '(bvslt x y)'
        >>> ULT(x, y).sexpr()
        '(bvult x y)'
        """
        a, b = _coerce_exprs(self, other)
        return BoolRef(Z3_mk_bvslt(self.ctx_ref(), a.as_ast(), b.as_ast()), self.ctx)

    def __gt__(self, other):
        """Create the Z3 expression (signed) `other > self`.

        Use the function UGT() for unsigned greater than.

        >>> x, y = BitVecs('x y', 32)
        >>> x > y
        x > y
        >>> (x > y).sexpr()
        '(bvsgt x y)'
        >>> UGT(x, y).sexpr()
        '(bvugt x y)'
        """
        a, b = _coerce_exprs(self, other)
        return BoolRef(Z3_mk_bvsgt(self.ctx_ref(), a.as_ast(), b.as_ast()), self.ctx)

    def __ge__(self, other):
        """Create the Z3 expression (signed) `other >= self`.

        Use the function UGE() for unsigned greater than or equal to.

        >>> x, y = BitVecs('x y', 32)
        >>> x >= y
        x >= y
        >>> (x >= y).sexpr()
        '(bvsge x y)'
        >>> UGE(x, y).sexpr()
        '(bvuge x y)'
        """
        a, b = _coerce_exprs(self, other)
        return BoolRef(Z3_mk_bvsge(self.ctx_ref(), a.as_ast(), b.as_ast()), self.ctx)

    def __rshift__(self, other):
        """Create the Z3 expression (arithmetical) right shift `self >> other`

        Use the function LShR() for the right logical shift

        >>> x, y = BitVecs('x y', 32)
        >>> x >> y
        x >> y
        >>> (x >> y).sexpr()
        '(bvashr x y)'
        >>> LShR(x, y).sexpr()
        '(bvlshr x y)'
        >>> BitVecVal(4, 3)
        4
        >>> BitVecVal(4, 3).as_signed_long()
        -4
        >>> simplify(BitVecVal(4, 3) >> 1).as_signed_long()
        -2
        >>> simplify(BitVecVal(4, 3) >> 1)
        6
        >>> simplify(LShR(BitVecVal(4, 3), 1))
        2
        >>> simplify(BitVecVal(2, 3) >> 1)
        1
        >>> simplify(LShR(BitVecVal(2, 3), 1))
        1
        """
        a, b = _coerce_exprs(self, other)
        return BitVecRef(Z3_mk_bvashr(self.ctx_ref(), a.as_ast(), b.as_ast()), self.ctx)

    def __lshift__(self, other):
        """Create the Z3 expression left shift `self << other`

        >>> x, y = BitVecs('x y', 32)
        >>> x << y
        x << y
        >>> (x << y).sexpr()
        '(bvshl x y)'
        >>> simplify(BitVecVal(2, 3) << 1)
        4
        """
        a, b = _coerce_exprs(self, other)
        return BitVecRef(Z3_mk_bvshl(self.ctx_ref(), a.as_ast(), b.as_ast()), self.ctx)

    def __rrshift__(self, other):
        """Create the Z3 expression (arithmetical) right shift `other` >> `self`.

        Use the function LShR() for the right logical shift

        >>> x = BitVec('x', 32)
        >>> 10 >> x
        10 >> x
        >>> (10 >> x).sexpr()
        '(bvashr #x0000000a x)'
        """
        a, b = _coerce_exprs(self, other)
        return BitVecRef(Z3_mk_bvashr(self.ctx_ref(), b.as_ast(), a.as_ast()), self.ctx)

    def __rlshift__(self, other):
        """Create the Z3 expression left shift `other << self`.

        Use the function LShR() for the right logical shift

        >>> x = BitVec('x', 32)
        >>> 10 << x
        10 << x
        >>> (10 << x).sexpr()
        '(bvshl #x0000000a x)'
        """
        a, b = _coerce_exprs(self, other)
        return BitVecRef(Z3_mk_bvshl(self.ctx_ref(), b.as_ast(), a.as_ast()), self.ctx)