from typing import Optional
from sympy.core.singleton import S
from sympy.assumptions import AppliedPredicate, ask, Predicate, Q  # type: ignore
from sympy.core.kind import BooleanKind
from sympy.core.relational import Eq, Ne, Gt, Lt, Ge, Le
from sympy.logic.boolalg import conjuncts, Not
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