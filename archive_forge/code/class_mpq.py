import operator
import sys
from .libmp import int_types, mpf_hash, bitcount, from_man_exp, HASH_MODULUS
class mpq(object):
    """
    Exact rational type, currently only intended for internal use.
    """
    __slots__ = ['_mpq_']

    def __new__(cls, p, q=1):
        if type(p) is tuple:
            p, q = p
        elif hasattr(p, '_mpq_'):
            p, q = p._mpq_
        return create_reduced(p, q)

    def __repr__(s):
        return 'mpq(%s,%s)' % s._mpq_

    def __str__(s):
        return '(%s/%s)' % s._mpq_

    def __int__(s):
        a, b = s._mpq_
        return a // b

    def __nonzero__(s):
        return bool(s._mpq_[0])
    __bool__ = __nonzero__

    def __hash__(s):
        a, b = s._mpq_
        if sys.version_info >= (3, 2):
            inverse = pow(b, HASH_MODULUS - 2, HASH_MODULUS)
            if not inverse:
                h = sys.hash_info.inf
            else:
                h = abs(a) * inverse % HASH_MODULUS
            if a < 0:
                h = -h
            if h == -1:
                h = -2
            return h
        else:
            if b == 1:
                return hash(a)
            if not b & b - 1:
                return mpf_hash(from_man_exp(a, 1 - bitcount(b)))
            return hash((a, b))

    def __eq__(s, t):
        ttype = type(t)
        if ttype is mpq:
            return s._mpq_ == t._mpq_
        if ttype in int_types:
            a, b = s._mpq_
            if b != 1:
                return False
            return a == t
        return NotImplemented

    def __ne__(s, t):
        ttype = type(t)
        if ttype is mpq:
            return s._mpq_ != t._mpq_
        if ttype in int_types:
            a, b = s._mpq_
            if b != 1:
                return True
            return a != t
        return NotImplemented

    def _cmp(s, t, op):
        ttype = type(t)
        if ttype in int_types:
            a, b = s._mpq_
            return op(a, t * b)
        if ttype is mpq:
            a, b = s._mpq_
            c, d = t._mpq_
            return op(a * d, b * c)
        return NotImplementedError

    def __lt__(s, t):
        return s._cmp(t, operator.lt)

    def __le__(s, t):
        return s._cmp(t, operator.le)

    def __gt__(s, t):
        return s._cmp(t, operator.gt)

    def __ge__(s, t):
        return s._cmp(t, operator.ge)

    def __abs__(s):
        a, b = s._mpq_
        if a >= 0:
            return s
        v = new(mpq)
        v._mpq_ = (-a, b)
        return v

    def __neg__(s):
        a, b = s._mpq_
        v = new(mpq)
        v._mpq_ = (-a, b)
        return v

    def __pos__(s):
        return s

    def __add__(s, t):
        ttype = type(t)
        if ttype is mpq:
            a, b = s._mpq_
            c, d = t._mpq_
            return create_reduced(a * d + b * c, b * d)
        if ttype in int_types:
            a, b = s._mpq_
            v = new(mpq)
            v._mpq_ = (a + b * t, b)
            return v
        return NotImplemented
    __radd__ = __add__

    def __sub__(s, t):
        ttype = type(t)
        if ttype is mpq:
            a, b = s._mpq_
            c, d = t._mpq_
            return create_reduced(a * d - b * c, b * d)
        if ttype in int_types:
            a, b = s._mpq_
            v = new(mpq)
            v._mpq_ = (a - b * t, b)
            return v
        return NotImplemented

    def __rsub__(s, t):
        ttype = type(t)
        if ttype is mpq:
            a, b = s._mpq_
            c, d = t._mpq_
            return create_reduced(b * c - a * d, b * d)
        if ttype in int_types:
            a, b = s._mpq_
            v = new(mpq)
            v._mpq_ = (b * t - a, b)
            return v
        return NotImplemented

    def __mul__(s, t):
        ttype = type(t)
        if ttype is mpq:
            a, b = s._mpq_
            c, d = t._mpq_
            return create_reduced(a * c, b * d)
        if ttype in int_types:
            a, b = s._mpq_
            return create_reduced(a * t, b)
        return NotImplemented
    __rmul__ = __mul__

    def __div__(s, t):
        ttype = type(t)
        if ttype is mpq:
            a, b = s._mpq_
            c, d = t._mpq_
            return create_reduced(a * d, b * c)
        if ttype in int_types:
            a, b = s._mpq_
            return create_reduced(a, b * t)
        return NotImplemented

    def __rdiv__(s, t):
        ttype = type(t)
        if ttype is mpq:
            a, b = s._mpq_
            c, d = t._mpq_
            return create_reduced(b * c, a * d)
        if ttype in int_types:
            a, b = s._mpq_
            return create_reduced(b * t, a)
        return NotImplemented

    def __pow__(s, t):
        ttype = type(t)
        if ttype in int_types:
            a, b = s._mpq_
            if t:
                if t < 0:
                    a, b, t = (b, a, -t)
                v = new(mpq)
                v._mpq_ = (a ** t, b ** t)
                return v
            raise ZeroDivisionError
        return NotImplemented