from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.printing.pretty.stringpict import prettyForm
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.operator import Operator
def _expand_pow(self, A, B, sign):
    exp = A.exp
    if not exp.is_integer or not exp.is_constant() or abs(exp) <= 1:
        return self
    base = A.base
    if exp.is_negative:
        base = A.base ** (-1)
        exp = -exp
    comm = Commutator(base, B).expand(commutator=True)
    result = base ** (exp - 1) * comm
    for i in range(1, exp):
        result += base ** (exp - 1 - i) * comm * base ** i
    return sign * result.expand()