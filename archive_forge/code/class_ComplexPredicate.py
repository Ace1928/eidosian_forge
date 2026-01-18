from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher
class ComplexPredicate(Predicate):
    """
    Complex number predicate.

    Explanation
    ===========

    ``Q.complex(x)`` is true iff ``x`` belongs to the set of complex
    numbers. Note that every complex number is finite.

    Examples
    ========

    >>> from sympy import Q, Symbol, ask, I, oo
    >>> x = Symbol('x')
    >>> ask(Q.complex(0))
    True
    >>> ask(Q.complex(2 + 3*I))
    True
    >>> ask(Q.complex(oo))
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Complex_number

    """
    name = 'complex'
    handler = Dispatcher('ComplexHandler', doc='Handler for Q.complex.\n\nTest that an expression belongs to the field of complex numbers.')