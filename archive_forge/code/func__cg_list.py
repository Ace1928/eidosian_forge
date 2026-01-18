from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import expand
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Wild, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.printing.pretty.stringpict import prettyForm, stringPict
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.physics.wigner import clebsch_gordan, wigner_3j, wigner_6j, wigner_9j
from sympy.printing.precedence import PRECEDENCE
def _cg_list(term):
    if isinstance(term, CG):
        return ((term,), 1, 1)
    cg = []
    coeff = 1
    if not isinstance(term, (Mul, Pow)):
        raise NotImplementedError('term must be CG, Add, Mul or Pow')
    if isinstance(term, Pow) and term.exp.is_number:
        if term.exp.is_number:
            [cg.append(term.base) for _ in range(term.exp)]
        else:
            return ((term,), 1, 1)
    if isinstance(term, Mul):
        for arg in term.args:
            if isinstance(arg, CG):
                cg.append(arg)
            else:
                coeff *= arg
        return (cg, coeff, coeff / abs(coeff))