import operator
from functools import reduce
from ..math_basics import prod
from ..sage_helper import _within_sage, sage_method, SageNotAvailable
from .realAlgebra import field_containing_real_and_imaginary_part_of_number_field
class SqrtLinCombination:
    """
    A class representing a linear combination

        c_1 * sqrt(r_1) + c_2 * sqrt(r_2) + ... + c_n * sqrt(r_n)

    where c_i and r_i have to be of type Integer, Rational or elements
    of the same Sage NumberField with a real embedding (Caution: this is
    assumed but not checked!) such that all r_i are positive (Caution: this is
    not checked during construction!).

    It implements +, -, * where one of the operators is allowed to be an
    integer or rational.

    / is only implemented when the denominator has only one term c_1 * sqrt(1).
    sqrt is only implemented for c_1 * sqrt(1) and it is not checked that
    c_1 is positive.

    == is implemented, but the other comparison operators are not: casting to
    a RealIntervalField is implemented instead and the user can compare the
    intervals.

    The == operator is implemented by first reducing A == B to D == 0 and then
    converting to a different data type (_FactorizedSqrtLinCombination) that can
    represent linear combinations::

      D =     c_1 * sqrt(r_{1,1}) * sqrt(r_{1,2}) * ... * sqrt(r_{1,k_1})
            + c_2 * sqrt(r_{2,1}) * sqrt(r_{2,2}) * ... * sqrt(r_{2,k_2})
            + ...
            + c_n * sqrt(r_{n,1}) * sqrt(r_{n,2}) * ... * sqrt(r_{n,k_n})

    by just trivially setting
       k_i = 0                       when r_i = 1 and
       r_{i,1} = r_i and k_1 = 1     otherwise.

    For this data type, multiplying two sqrt(r_{i,j}) with equal r_{i,j} will
    cancel the two sqrt's and apply the common r_{i,j} to the c_i of the result
    instead. Thus, the following procedure for determining whether D == 0 will
    eventually terminate:

    - if the number of terms n is 0, return True
    - if the number of terms n is 1, return c_1 == 0
    - if there is a r_{i,j} common to each summand, factor it out
    - pick one of the r_{i,j}, split the sum into two parts "left",
      respectively, "right" of all the terms containing sqrt(r_{i,j}),
      respectively, not containing sqrt(r_{i,j}).
    - If left^2 - right^2 == 0 is False, return False.
      (sqrt(r_{i,j})^2 simplifies to r_{i,j} and disappears, so the resulting
      expression is easier and this recursion terminates eventually.)
    - If left == 0 (some comment applies), return True
    - Use interval arithmetic of increasing precision until it is high enough
      to determine the signs of left and right.
      Return True if and only if the signs differ, otherwise False.

    Examples::

        sage: from sage.rings.number_field.number_field import NumberField
        sage: from sage.rings.integer import Integer
        sage: from sage.rings.rational import Rational
        sage: from sage.rings.real_mpfr import RealLiteral, RealField
        sage: from sage.rings.real_mpfi import RealIntervalField
        sage: from sage.calculus.var import var
        sage: from sage.functions.other import sqrt
        sage: x = var('x')
        sage: poly = x ** 6 + Rational((3,2))*x**4 + Rational((9,16))*x**2 - Rational((23,64))
        sage: nf = NumberField(poly, 'z', embedding = RealField()(0.56227951206))
        sage: z = nf.gen()

        sage: A = SqrtLinCombination(z)
        sage: B = SqrtLinCombination(Rational((8,9))*z**4 + Rational((10,9))*z**2 + Rational((2,9)))
        sage: C = SqrtLinCombination(3)
        sage: D = SqrtLinCombination(Integer(5))
        sage: E = SqrtLinCombination(Rational((6,7)))

        sage: A + B
        (8/9*z^4 + 10/9*z^2 + z + 2/9) * sqrt(1)
        sage: B - E
        (8/9*z^4 + 10/9*z^2 - 40/63) * sqrt(1)
        sage: A + sqrt(B) * sqrt(B)
        (8/9*z^4 + 10/9*z^2 + z + 2/9) * sqrt(1)
        sage: A + sqrt(B) * sqrt(B) + C == A + B + C
        True
        sage: A / E
        (7/6*z) * sqrt(1)
        sage: B / A.sqrt()
        (128/207*z^5 + 376/207*z^3 + 302/207*z) * sqrt(z)
        sage: B / (D * A.sqrt())
        (128/1035*z^5 + 376/1035*z^3 + 302/1035*z) * sqrt(z)
        sage: RIF = RealIntervalField(100)
        sage: RIF(B.sqrt() + E.sqrt())
        1.73967449622339881238507307209?
        sage: A - B == 0
        False
        sage: (A + B).sqrt()
        (1) * sqrt(8/9*z^4 + 10/9*z^2 + z + 2/9)
        sage: 3 * A.sqrt() + (4 * B).sqrt() + C + 8 == (9 * A).sqrt() + 2 * B.sqrt() + (C * C).sqrt() + 11 - 3
        True

    """

    def __init__(self, value=None, d={}, embed_cache=None):
        if value is not None:
            if d:
                raise TypeError('SqrtLinCombination has both value and dictionary.')
            self._dict = _filter_zero({_One: _convert_to_allowed_type(value)})
        else:
            self._dict = _filter_zero(d)
        self._embed_cache = embed_cache

    def __add__(self, other):
        if not isinstance(other, SqrtLinCombination):
            return self + SqrtLinCombination(other, embed_cache=_get_embed_cache(self, other))
        d = {}
        for k, v in self._dict.items():
            d[k] = d.get(k, 0) + v
        for k, v in other._dict.items():
            d[k] = d.get(k, 0) + v
        return SqrtLinCombination(d=d, embed_cache=_get_embed_cache(self, other))

    def __neg__(self):
        return SqrtLinCombination(d=dict(((k, -v) for k, v in self._dict.items())), embed_cache=self._embed_cache)

    def __sub__(self, other):
        return self + -other

    def __mul__(self, other):
        if not isinstance(other, SqrtLinCombination):
            return self * SqrtLinCombination(other, embed_cache=_get_embed_cache(self, other))
        d = {}
        for k1, v1 in self._dict.items():
            for k2, v2 in other._dict.items():
                p = v1 * v2
                if k1 == k2:
                    d[_One] = d.get(_One, 0) + k1 * p
                else:
                    m = k1 * k2
                    d[m] = d.get(m, 0) + p
        return SqrtLinCombination(d=d, embed_cache=_get_embed_cache(self, other))

    def inverse(self):
        l = len(self._dict)
        if l != 1:
            if l == 0:
                raise ZeroDivisionError('SqrtLinCombination division by zero')
            raise TypeError('SqrtLinCombination division not fully implemented')
        for k, v in self._dict.items():
            return SqrtLinCombination(d={k: 1 / (v * k)}, embed_cache=self._embed_cache)

    def __div__(self, other):
        if not isinstance(other, SqrtLinCombination):
            return self / SqrtLinCombination(other, embed_cache=_get_embed_cache(self, other))
        return self * other.inverse()

    def __truediv__(self, other):
        return self.__div__(other)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return -self + other

    def __rmul__(self, other):
        return self * other

    def __rdiv__(self, other):
        return self.inverse() * other

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    def sqrt(self):
        l = len(self._dict)
        if l == 0:
            return SqrtLinCombination(embed_cache=self._embed_cache)
        if l == 1:
            for k, v in self._dict.items():
                if k != 1:
                    raise TypeError('SqrtLinCombination sqrt not fully implemented')
                return SqrtLinCombination(d={v: _One}, embed_cache=self._embed_cache)
        raise TypeError('SqrtLinCombination sqrt not fully implemented')

    def __repr__(self):
        if self._dict:
            return '+'.join(['(%r) * sqrt(%r)' % (v, k) for k, v in self._dict.items()])
        return '0'

    def __eq__(self, other):
        """
        Implements the == operator as described above.
        """
        diff = self - other
        f = _FactorizedSqrtLinCombination.from_sqrt_lin_combination(diff)
        return f.is_zero()

    def __lt__(self, other):
        raise Exception('Not implemented')

    def __le__(self, other):
        raise Exception('Not implemented')

    def __gt__(self, other):
        raise Exception('Not implemented')

    def __ge__(self, other):
        raise Exception('Not implemented')

    def _real_mpfi_(self, RIF):
        """
        Convert to interval in given RealIntervalField instance.
        """

        def eval_term(k, v):
            s = _to_RIF(k, RIF, self._embed_cache)
            if not s > 0:
                raise _SqrtException()
            return _to_RIF(v, RIF, self._embed_cache) * s.sqrt()
        return sum([eval_term(k, v) for k, v in self._dict.items()], RIF(0))

    def _sign_numerical(self, prec):
        """
        Use interval arithmetics with precision prec to try to determine the
        sign. If we could not certify the sign, return None.
        The result is a pair (sign, interval).
        """
        RIF = RealIntervalField(prec)
        try:
            interval_val = RIF(self)
        except _SqrtException:
            return (None, None)
        if interval_val > 0:
            return (+1, interval_val)
        if interval_val < 0:
            return (-1, interval_val)
        return (None, interval_val)

    def sign_with_interval(self):
        """
        Similar to sign, but for the non-zero case, also return the interval
        certifying the sign - useful for debugging.
        """
        prec = 106
        numerical_sign, interval_val = self._sign_numerical(prec)
        if numerical_sign is not None:
            return (numerical_sign, interval_val)
        if self == 0:
            return (0, 0)
        while True:
            prec *= 2
            numerical_sign, interval_val = self._sign_numerical(prec)
            if numerical_sign is not None:
                return (numerical_sign, interval_val)

    def sign(self):
        """
        Returns the +1, 0, -1 depending on whether the value is positive,
        zero or negative. For the zero case, exact arithmetic is used to
        certify. Otherwise, interval arithmetic is used.
        """
        return self.sign_with_interval()[0]