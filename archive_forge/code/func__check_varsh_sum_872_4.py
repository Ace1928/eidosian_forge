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
def _check_varsh_sum_872_4(e):
    alpha = symbols('alpha')
    beta = symbols('beta')
    a = Wild('a')
    b = Wild('b')
    c = Wild('c')
    cp = Wild('cp')
    gamma = Wild('gamma')
    gammap = Wild('gammap')
    cg1 = CG(a, alpha, b, beta, c, gamma)
    cg2 = CG(a, alpha, b, beta, cp, gammap)
    match1 = e.match(Sum(cg1 * cg2, (alpha, -a, a), (beta, -b, b)))
    if match1 is not None and len(match1) == 6:
        return (KroneckerDelta(c, cp) * KroneckerDelta(gamma, gammap)).subs(match1)
    match2 = e.match(Sum(cg1 ** 2, (alpha, -a, a), (beta, -b, b)))
    if match2 is not None and len(match2) == 4:
        return S.One
    return e