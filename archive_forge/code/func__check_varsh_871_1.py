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
def _check_varsh_871_1(term_list):
    a, alpha, b, lt = map(Wild, ('a', 'alpha', 'b', 'lt'))
    expr = lt * CG(a, alpha, b, 0, a, alpha)
    simp = (2 * a + 1) * KroneckerDelta(b, 0)
    sign = lt / abs(lt)
    build_expr = 2 * a + 1
    index_expr = a + alpha
    return _check_cg_simp(expr, simp, sign, lt, term_list, (a, alpha, b, lt), (a, b), build_expr, index_expr)