from sympy.assumptions import Q
from sympy.core.relational import is_eq, is_neq, is_gt, is_ge, is_lt, is_le
from .binrel import BinaryRelation
class StrictGreaterThanPredicate(BinaryRelation):
    """
    Binary predicate for $>$.

    The purpose of this class is to provide the instance which represent
    the ">" predicate in order to allow the logical inference.
    This class must remain internal to assumptions module and user must
    use :obj:`~.Gt()` instead to construct the equality expression.

    Evaluating this predicate to ``True`` or ``False`` is done by
    :func:`~.core.relational.is_gt()`

    Examples
    ========

    >>> from sympy import ask, Q
    >>> Q.gt(0, 0)
    Q.gt(0, 0)
    >>> ask(_)
    False

    See Also
    ========

    sympy.core.relational.Gt

    """
    is_reflexive = False
    is_symmetric = False
    name = 'gt'
    handler = None

    @property
    def reversed(self):
        return Q.lt

    @property
    def negated(self):
        return Q.le

    def eval(self, args, assumptions=True):
        if assumptions == True:
            assumptions = None
        return is_gt(*args, assumptions)