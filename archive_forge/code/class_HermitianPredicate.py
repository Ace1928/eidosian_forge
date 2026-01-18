from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher
class HermitianPredicate(Predicate):
    """
    Hermitian predicate.

    Explanation
    ===========

    ``ask(Q.hermitian(x))`` is true iff ``x`` belongs to the set of
    Hermitian operators.

    References
    ==========

    .. [1] https://mathworld.wolfram.com/HermitianOperator.html

    """
    name = 'hermitian'
    handler = Dispatcher('HermitianHandler', doc='Handler for Q.hermitian.\n\nTest that an expression belongs to the field of Hermitian operators.')