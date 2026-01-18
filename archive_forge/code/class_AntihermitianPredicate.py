from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher
class AntihermitianPredicate(Predicate):
    """
    Antihermitian predicate.

    Explanation
    ===========

    ``Q.antihermitian(x)`` is true iff ``x`` belongs to the field of
    antihermitian operators, i.e., operators in the form ``x*I``, where
    ``x`` is Hermitian.

    References
    ==========

    .. [1] https://mathworld.wolfram.com/HermitianOperator.html

    """
    name = 'antihermitian'
    handler = Dispatcher('AntiHermitianHandler', doc='Handler for Q.antihermitian.\n\nTest that an expression belongs to the field of anti-Hermitian\noperators, that is, operators in the form x*I, where x is Hermitian.')