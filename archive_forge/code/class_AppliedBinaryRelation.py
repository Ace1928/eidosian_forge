from typing import Optional
from sympy.core.singleton import S
from sympy.assumptions import AppliedPredicate, ask, Predicate, Q  # type: ignore
from sympy.core.kind import BooleanKind
from sympy.core.relational import Eq, Ne, Gt, Lt, Ge, Le
from sympy.logic.boolalg import conjuncts, Not
class AppliedBinaryRelation(AppliedPredicate):
    """
    The class of expressions resulting from applying ``BinaryRelation``
    to the arguments.

    """

    @property
    def lhs(self):
        """The left-hand side of the relation."""
        return self.arguments[0]

    @property
    def rhs(self):
        """The right-hand side of the relation."""
        return self.arguments[1]

    @property
    def reversed(self):
        """
        Try to return the relationship with sides reversed.
        """
        revfunc = self.function.reversed
        if revfunc is None:
            return self
        return revfunc(self.rhs, self.lhs)

    @property
    def reversedsign(self):
        """
        Try to return the relationship with signs reversed.
        """
        revfunc = self.function.reversed
        if revfunc is None:
            return self
        if not any((side.kind is BooleanKind for side in self.arguments)):
            return revfunc(-self.lhs, -self.rhs)
        return self

    @property
    def negated(self):
        neg_rel = self.function.negated
        if neg_rel is None:
            return Not(self, evaluate=False)
        return neg_rel(*self.arguments)

    def _eval_ask(self, assumptions):
        conj_assumps = set()
        binrelpreds = {Eq: Q.eq, Ne: Q.ne, Gt: Q.gt, Lt: Q.lt, Ge: Q.ge, Le: Q.le}
        for a in conjuncts(assumptions):
            if a.func in binrelpreds:
                conj_assumps.add(binrelpreds[type(a)](*a.args))
            else:
                conj_assumps.add(a)
        if any((rel in conj_assumps for rel in (self, self.reversed))):
            return True
        neg_rels = (self.negated, self.reversed.negated, Not(self, evaluate=False), Not(self.reversed, evaluate=False))
        if any((rel in conj_assumps for rel in neg_rels)):
            return False
        ret = self.function.eval(self.arguments, assumptions)
        if ret is not None:
            return ret
        args = tuple((a.simplify() for a in self.arguments))
        return self.function.eval(args, assumptions)

    def __bool__(self):
        ret = ask(self)
        if ret is None:
            raise TypeError('Cannot determine truth value of %s' % self)
        return ret