from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher
class ImaginaryPredicate(Predicate):
    """
    Imaginary number predicate.

    Explanation
    ===========

    ``Q.imaginary(x)`` is true iff ``x`` can be written as a real
    number multiplied by the imaginary unit ``I``. Please note that ``0``
    is not considered to be an imaginary number.

    Examples
    ========

    >>> from sympy import Q, ask, I
    >>> ask(Q.imaginary(3*I))
    True
    >>> ask(Q.imaginary(2 + 3*I))
    False
    >>> ask(Q.imaginary(0))
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Imaginary_number

    """
    name = 'imaginary'
    handler = Dispatcher('ImaginaryHandler', doc='Handler for Q.imaginary.\n\nTest that an expression belongs to the field of imaginary numbers,\nthat is, numbers in the form x*I, where x is real.')