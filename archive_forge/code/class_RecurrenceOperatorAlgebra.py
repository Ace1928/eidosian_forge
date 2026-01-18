from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.printing import sstr
from sympy.core.sympify import sympify
class RecurrenceOperatorAlgebra:
    """
    A Recurrence Operator Algebra is a set of noncommutative polynomials
    in intermediate `Sn` and coefficients in a base ring A. It follows the
    commutation rule:
    Sn * a(n) = a(n + 1) * Sn

    This class represents a Recurrence Operator Algebra and serves as the parent ring
    for Recurrence Operators.

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy import symbols
    >>> from sympy.holonomic.recurrence import RecurrenceOperators
    >>> n = symbols('n', integer=True)
    >>> R, Sn = RecurrenceOperators(ZZ.old_poly_ring(n), 'Sn')
    >>> R
    Univariate Recurrence Operator Algebra in intermediate Sn over the base ring
    ZZ[n]

    See Also
    ========

    RecurrenceOperator
    """

    def __init__(self, base, generator):
        self.base = base
        self.shift_operator = RecurrenceOperator([base.zero, base.one], self)
        if generator is None:
            self.gen_symbol = symbols('Sn', commutative=False)
        elif isinstance(generator, str):
            self.gen_symbol = symbols(generator, commutative=False)
        elif isinstance(generator, Symbol):
            self.gen_symbol = generator

    def __str__(self):
        string = 'Univariate Recurrence Operator Algebra in intermediate ' + sstr(self.gen_symbol) + ' over the base ring ' + self.base.__str__()
        return string
    __repr__ = __str__

    def __eq__(self, other):
        if self.base == other.base and self.gen_symbol == other.gen_symbol:
            return True
        else:
            return False