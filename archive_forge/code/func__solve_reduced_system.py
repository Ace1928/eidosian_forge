import itertools
from sympy.core import S
from sympy.core.sorting import default_sort_key
from sympy.polys import Poly, groebner, roots
from sympy.polys.polytools import parallel_poly_from_expr
from sympy.polys.polyerrors import (ComputationFailed,
from sympy.simplify import rcollect
from sympy.utilities import postfixes
from sympy.utilities.misc import filldedent
def _solve_reduced_system(system, gens, entry=False):
    """Recursively solves reduced polynomial systems. """
    if len(system) == len(gens) == 1:
        zeros = list(roots(system[0], gens[-1], strict=strict).keys())
        return [(zero,) for zero in zeros]
    basis = groebner(system, gens, polys=True)
    if len(basis) == 1 and basis[0].is_ground:
        if not entry:
            return []
        else:
            return None
    univariate = list(filter(_is_univariate, basis))
    if len(basis) < len(gens):
        raise NotImplementedError(filldedent('\n                only zero-dimensional systems supported\n                (finite number of solutions)\n                '))
    if len(univariate) == 1:
        f = univariate.pop()
    else:
        raise NotImplementedError(filldedent('\n                only zero-dimensional systems supported\n                (finite number of solutions)\n                '))
    gens = f.gens
    gen = gens[-1]
    zeros = list(roots(f.ltrim(gen), strict=strict).keys())
    if not zeros:
        return []
    if len(basis) == 1:
        return [(zero,) for zero in zeros]
    solutions = []
    for zero in zeros:
        new_system = []
        new_gens = gens[:-1]
        for b in basis[:-1]:
            eq = _subs_root(b, gen, zero)
            if eq is not S.Zero:
                new_system.append(eq)
        for solution in _solve_reduced_system(new_system, new_gens):
            solutions.append(solution + (zero,))
    if solutions and len(solutions[0]) != len(gens):
        raise NotImplementedError(filldedent('\n                only zero-dimensional systems supported\n                (finite number of solutions)\n                '))
    return solutions