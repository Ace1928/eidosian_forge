from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.printing import sstr
from sympy.core.sympify import sympify
def _mul_dmp_diffop(b, listofother):
    if isinstance(listofother, list):
        sol = []
        for i in listofother:
            sol.append(i * b)
        return sol
    else:
        return [b * listofother]