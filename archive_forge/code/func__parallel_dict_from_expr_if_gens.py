from sympy.core import (S, Add, Mul, Pow, Eq, Expr,
from sympy.core.exprtools import decompose_power, decompose_power_rat
from sympy.core.numbers import _illegal
from sympy.polys.polyerrors import PolynomialError, GeneratorsError
from sympy.polys.polyoptions import build_options
import re
def _parallel_dict_from_expr_if_gens(exprs, opt):
    """Transform expressions into a multinomial form given generators. """
    k, indices = (len(opt.gens), {})
    for i, g in enumerate(opt.gens):
        indices[g] = i
    polys = []
    for expr in exprs:
        poly = {}
        if expr.is_Equality:
            expr = expr.lhs - expr.rhs
        for term in Add.make_args(expr):
            coeff, monom = ([], [0] * k)
            for factor in Mul.make_args(term):
                if not _not_a_coeff(factor) and factor.is_Number:
                    coeff.append(factor)
                else:
                    try:
                        if opt.series is False:
                            base, exp = decompose_power(factor)
                            if exp < 0:
                                exp, base = (-exp, Pow(base, -S.One))
                        else:
                            base, exp = decompose_power_rat(factor)
                        monom[indices[base]] = exp
                    except KeyError:
                        if not factor.has_free(*opt.gens):
                            coeff.append(factor)
                        else:
                            raise PolynomialError('%s contains an element of the set of generators.' % factor)
            monom = tuple(monom)
            if monom in poly:
                poly[monom] += Mul(*coeff)
            else:
                poly[monom] = Mul(*coeff)
        polys.append(poly)
    return (polys, opt.gens)