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
def _cg_simp_add(e):
    """Takes a sum of terms involving Clebsch-Gordan coefficients and
    simplifies the terms.

    Explanation
    ===========

    First, we create two lists, cg_part, which is all the terms involving CG
    coefficients, and other_part, which is all other terms. The cg_part list
    is then passed to the simplification methods, which return the new cg_part
    and any additional terms that are added to other_part
    """
    cg_part = []
    other_part = []
    e = expand(e)
    for arg in e.args:
        if arg.has(CG):
            if isinstance(arg, Sum):
                other_part.append(_cg_simp_sum(arg))
            elif isinstance(arg, Mul):
                terms = 1
                for term in arg.args:
                    if isinstance(term, Sum):
                        terms *= _cg_simp_sum(term)
                    else:
                        terms *= term
                if terms.has(CG):
                    cg_part.append(terms)
                else:
                    other_part.append(terms)
            else:
                cg_part.append(arg)
        else:
            other_part.append(arg)
    cg_part, other = _check_varsh_871_1(cg_part)
    other_part.append(other)
    cg_part, other = _check_varsh_871_2(cg_part)
    other_part.append(other)
    cg_part, other = _check_varsh_872_9(cg_part)
    other_part.append(other)
    return Add(*cg_part) + Add(*other_part)