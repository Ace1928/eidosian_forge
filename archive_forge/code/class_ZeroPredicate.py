from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher
class ZeroPredicate(Predicate):
    """
    Zero number predicate.

    Explanation
    ===========

    ``ask(Q.zero(x))`` is true iff the value of ``x`` is zero.

    Examples
    ========

    >>> from sympy import ask, Q, oo, symbols
    >>> x, y = symbols('x, y')
    >>> ask(Q.zero(0))
    True
    >>> ask(Q.zero(1/oo))
    True
    >>> print(ask(Q.zero(0*oo)))
    None
    >>> ask(Q.zero(1))
    False
    >>> ask(Q.zero(x*y), Q.zero(x) | Q.zero(y))
    True

    """
    name = 'zero'
    handler = Dispatcher('ZeroHandler', doc="Handler for key 'zero'.")