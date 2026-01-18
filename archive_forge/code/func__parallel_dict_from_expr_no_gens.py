from sympy.core import (S, Add, Mul, Pow, Eq, Expr,
from sympy.core.exprtools import decompose_power, decompose_power_rat
from sympy.core.numbers import _illegal
from sympy.polys.polyerrors import PolynomialError, GeneratorsError
from sympy.polys.polyoptions import build_options
import re
def _parallel_dict_from_expr_no_gens(exprs, opt):
    """Transform expressions into a multinomial form and figure out generators. """
    if opt.domain is not None:

        def _is_coeff(factor):
            return factor in opt.domain
    elif opt.extension is True:

        def _is_coeff(factor):
            return factor.is_algebraic
    elif opt.greedy is not False:

        def _is_coeff(factor):
            return factor is S.ImaginaryUnit
    else:

        def _is_coeff(factor):
            return factor.is_number
    gens, reprs = (set(), [])
    for expr in exprs:
        terms = []
        if expr.is_Equality:
            expr = expr.lhs - expr.rhs
        for term in Add.make_args(expr):
            coeff, elements = ([], {})
            for factor in Mul.make_args(term):
                if not _not_a_coeff(factor) and (factor.is_Number or _is_coeff(factor)):
                    coeff.append(factor)
                else:
                    if opt.series is False:
                        base, exp = decompose_power(factor)
                        if exp < 0:
                            exp, base = (-exp, Pow(base, -S.One))
                    else:
                        base, exp = decompose_power_rat(factor)
                    elements[base] = elements.setdefault(base, 0) + exp
                    gens.add(base)
            terms.append((coeff, elements))
        reprs.append(terms)
    gens = _sort_gens(gens, opt=opt)
    k, indices = (len(gens), {})
    for i, g in enumerate(gens):
        indices[g] = i
    polys = []
    for terms in reprs:
        poly = {}
        for coeff, term in terms:
            monom = [0] * k
            for base, exp in term.items():
                monom[indices[base]] = exp
            monom = tuple(monom)
            if monom in poly:
                poly[monom] += Mul(*coeff)
            else:
                poly[monom] = Mul(*coeff)
        polys.append(poly)
    return (polys, tuple(gens))