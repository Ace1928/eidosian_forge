from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.printing import sstr
from sympy.core.sympify import sympify
def _mul_Sni_b(b):
    sol = [base.zero]
    if isinstance(b, list):
        for i in b:
            j = base.to_sympy(i).subs(base.gens[0], base.gens[0] + S.One)
            sol.append(base.from_sympy(j))
    else:
        j = b.subs(base.gens[0], base.gens[0] + S.One)
        sol.append(base.from_sympy(j))
    return sol