import math
from sympy.core import S
from sympy.core.basic import Basic
from sympy.core.function import Function
from sympy.core.numbers import Integer
class SchurNumber(Function):
    """
    This function creates a SchurNumber object
    which is evaluated for `k \\le 5` otherwise only
    the lower bound information can be retrieved.

    Examples
    ========

    >>> from sympy.combinatorics.schur_number import SchurNumber

    Since S(3) = 13, hence the output is a number
    >>> SchurNumber(3)
    13

    We do not know the Schur number for values greater than 5, hence
    only the object is returned
    >>> SchurNumber(6)
    SchurNumber(6)

    Now, the lower bound information can be retrieved using lower_bound()
    method
    >>> SchurNumber(6).lower_bound()
    536

    """

    @classmethod
    def eval(cls, k):
        if k.is_Number:
            if k is S.Infinity:
                return S.Infinity
            if k.is_zero:
                return S.Zero
            if not k.is_integer or k.is_negative:
                raise ValueError('k should be a positive integer')
            first_known_schur_numbers = {1: 1, 2: 4, 3: 13, 4: 44, 5: 160}
            if k <= 5:
                return Integer(first_known_schur_numbers[k])

    def lower_bound(self):
        f_ = self.args[0]
        if f_ == 6:
            return Integer(536)
        if f_ == 7:
            return Integer(1680)
        if f_.is_Integer:
            return 3 * self.func(f_ - 1).lower_bound() - 1
        return (3 ** f_ - 1) / 2