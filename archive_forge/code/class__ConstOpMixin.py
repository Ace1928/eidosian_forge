import functools
import string
import re
from types import MappingProxyType
from llvmlite.ir import values, types, _utils
from llvmlite.ir._utils import (_StrCaching, _StringReferenceCaching,
class _ConstOpMixin(object):
    """
    A mixin defining constant operations, for use in constant-like classes.
    """

    @_binop('shl')
    def shl(self, other):
        """
        Left integer shift:
            lhs << rhs
        """

    @_binop('lshr')
    def lshr(self, other):
        """
        Logical (unsigned) right integer shift:
            lhs >> rhs
        """

    @_binop('ashr')
    def ashr(self, other):
        """
        Arithmetic (signed) right integer shift:
            lhs >> rhs
        """

    @_binop('add')
    def add(self, other):
        """
        Integer addition:
            lhs + rhs
        """

    @_binop('fadd')
    def fadd(self, other):
        """
        Floating-point addition:
            lhs + rhs
        """

    @_binop('sub')
    def sub(self, other):
        """
        Integer subtraction:
            lhs - rhs
        """

    @_binop('fsub')
    def fsub(self, other):
        """
        Floating-point subtraction:
            lhs - rhs
        """

    @_binop('mul')
    def mul(self, other):
        """
        Integer multiplication:
            lhs * rhs
        """

    @_binop('fmul')
    def fmul(self, other):
        """
        Floating-point multiplication:
            lhs * rhs
        """

    @_binop('udiv')
    def udiv(self, other):
        """
        Unsigned integer division:
            lhs / rhs
        """

    @_binop('sdiv')
    def sdiv(self, other):
        """
        Signed integer division:
            lhs / rhs
        """

    @_binop('fdiv')
    def fdiv(self, other):
        """
        Floating-point division:
            lhs / rhs
        """

    @_binop('urem')
    def urem(self, other):
        """
        Unsigned integer remainder:
            lhs % rhs
        """

    @_binop('srem')
    def srem(self, other):
        """
        Signed integer remainder:
            lhs % rhs
        """

    @_binop('frem')
    def frem(self, other):
        """
        Floating-point remainder:
            lhs % rhs
        """

    @_binop('or')
    def or_(self, other):
        """
        Bitwise integer OR:
            lhs | rhs
        """

    @_binop('and')
    def and_(self, other):
        """
        Bitwise integer AND:
            lhs & rhs
        """

    @_binop('xor')
    def xor(self, other):
        """
        Bitwise integer XOR:
            lhs ^ rhs
        """

    def _cmp(self, prefix, sign, cmpop, other):
        ins = prefix + 'cmp'
        try:
            op = _CMP_MAP[cmpop]
        except KeyError:
            raise ValueError('invalid comparison %r for %s' % (cmpop, ins))
        if not (prefix == 'i' and cmpop in ('==', '!=')):
            op = sign + op
        if self.type != other.type:
            raise ValueError('Operands must be the same type, got (%s, %s)' % (self.type, other.type))
        fmt = '{0} {1} ({2} {3}, {4} {5})'.format(ins, op, self.type, self.get_reference(), other.type, other.get_reference())
        return FormattedConstant(types.IntType(1), fmt)

    def icmp_signed(self, cmpop, other):
        """
        Signed integer comparison:
            lhs <cmpop> rhs

        where cmpop can be '==', '!=', '<', '<=', '>', '>='
        """
        return self._cmp('i', 's', cmpop, other)

    def icmp_unsigned(self, cmpop, other):
        """
        Unsigned integer (or pointer) comparison:
            lhs <cmpop> rhs

        where cmpop can be '==', '!=', '<', '<=', '>', '>='
        """
        return self._cmp('i', 'u', cmpop, other)

    def fcmp_ordered(self, cmpop, other):
        """
        Floating-point ordered comparison:
            lhs <cmpop> rhs

        where cmpop can be '==', '!=', '<', '<=', '>', '>=', 'ord', 'uno'
        """
        return self._cmp('f', 'o', cmpop, other)

    def fcmp_unordered(self, cmpop, other):
        """
        Floating-point unordered comparison:
            lhs <cmpop> rhs

        where cmpop can be '==', '!=', '<', '<=', '>', '>=', 'ord', 'uno'
        """
        return self._cmp('f', 'u', cmpop, other)

    def not_(self):
        """
        Bitwise integer complement:
            ~value
        """
        if isinstance(self.type, types.VectorType):
            rhs = values.Constant(self.type, (-1,) * self.type.count)
        else:
            rhs = values.Constant(self.type, -1)
        return self.xor(rhs)

    def neg(self):
        """
        Integer negative:
            -value
        """
        zero = values.Constant(self.type, 0)
        return zero.sub(self)

    def fneg(self):
        """
        Floating-point negative:
            -value
        """
        fmt = 'fneg ({0} {1})'.format(self.type, self.get_reference())
        return FormattedConstant(self.type, fmt)

    @_castop('trunc')
    def trunc(self, typ):
        """
        Truncating integer downcast to a smaller type.
        """

    @_castop('zext')
    def zext(self, typ):
        """
        Zero-extending integer upcast to a larger type
        """

    @_castop('sext')
    def sext(self, typ):
        """
        Sign-extending integer upcast to a larger type.
        """

    @_castop('fptrunc')
    def fptrunc(self, typ):
        """
        Floating-point downcast to a less precise type.
        """

    @_castop('fpext')
    def fpext(self, typ):
        """
        Floating-point upcast to a more precise type.
        """

    @_castop('bitcast')
    def bitcast(self, typ):
        """
        Pointer cast to a different pointer type.
        """

    @_castop('fptoui')
    def fptoui(self, typ):
        """
        Convert floating-point to unsigned integer.
        """

    @_castop('uitofp')
    def uitofp(self, typ):
        """
        Convert unsigned integer to floating-point.
        """

    @_castop('fptosi')
    def fptosi(self, typ):
        """
        Convert floating-point to signed integer.
        """

    @_castop('sitofp')
    def sitofp(self, typ):
        """
        Convert signed integer to floating-point.
        """

    @_castop('ptrtoint')
    def ptrtoint(self, typ):
        """
        Cast pointer to integer.
        """
        if not isinstance(self.type, types.PointerType):
            msg = "can only call ptrtoint() on pointer type, not '%s'"
            raise TypeError(msg % (self.type,))
        if not isinstance(typ, types.IntType):
            raise TypeError("can only ptrtoint() to integer type, not '%s'" % (typ,))

    @_castop('inttoptr')
    def inttoptr(self, typ):
        """
        Cast integer to pointer.
        """
        if not isinstance(self.type, types.IntType):
            msg = "can only call inttoptr() on integer constants, not '%s'"
            raise TypeError(msg % (self.type,))
        if not isinstance(typ, types.PointerType):
            raise TypeError("can only inttoptr() to pointer type, not '%s'" % (typ,))

    def gep(self, indices):
        """
        Call getelementptr on this pointer constant.
        """
        if not isinstance(self.type, types.PointerType):
            raise TypeError("can only call gep() on pointer constants, not '%s'" % (self.type,))
        outtype = self.type
        for i in indices:
            outtype = outtype.gep(i)
        strindices = ['{0} {1}'.format(idx.type, idx.get_reference()) for idx in indices]
        op = 'getelementptr ({0}, {1} {2}, {3})'.format(self.type.pointee, self.type, self.get_reference(), ', '.join(strindices))
        return FormattedConstant(outtype.as_pointer(self.addrspace), op)