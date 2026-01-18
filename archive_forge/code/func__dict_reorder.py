from sympy.core import (S, Add, Mul, Pow, Eq, Expr,
from sympy.core.exprtools import decompose_power, decompose_power_rat
from sympy.core.numbers import _illegal
from sympy.polys.polyerrors import PolynomialError, GeneratorsError
from sympy.polys.polyoptions import build_options
import re
def _dict_reorder(rep, gens, new_gens):
    """Reorder levels using dict representation. """
    gens = list(gens)
    monoms = rep.keys()
    coeffs = rep.values()
    new_monoms = [[] for _ in range(len(rep))]
    used_indices = set()
    for gen in new_gens:
        try:
            j = gens.index(gen)
            used_indices.add(j)
            for M, new_M in zip(monoms, new_monoms):
                new_M.append(M[j])
        except ValueError:
            for new_M in new_monoms:
                new_M.append(0)
    for i, _ in enumerate(gens):
        if i not in used_indices:
            for monom in monoms:
                if monom[i]:
                    raise GeneratorsError('unable to drop generators')
    return (map(tuple, new_monoms), coeffs)