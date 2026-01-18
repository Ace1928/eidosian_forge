from sympy.core.add import Add
from sympy.core.assumptions import check_assumptions
from sympy.core.containers import Tuple
from sympy.core.exprtools import factor_terms
from sympy.core.function import _mexpand
from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.numbers import igcdex, ilcm, igcd
from sympy.core.power import integer_nthroot, isqrt
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Symbol, symbols
from sympy.core.sympify import _sympify
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.ntheory.factor_ import (
from sympy.ntheory.generate import nextprime
from sympy.ntheory.primetest import is_square, isprime
from sympy.ntheory.residue_ntheory import sqrt_mod
from sympy.polys.polyerrors import GeneratorsNeeded
from sympy.polys.polytools import Poly, factor_list
from sympy.simplify.simplify import signsimp
from sympy.solvers.solveset import solveset_real
from sympy.utilities import numbered_symbols
from sympy.utilities.misc import as_int, filldedent
from sympy.utilities.iterables import (is_sequence, subsets, permute_signs,
class GeneralPythagorean(DiophantineEquationType):
    """
    Representation of the general pythagorean equation,
    `a_{1}^2x_{1}^2 + a_{2}^2x_{2}^2 + . . . + a_{n}^2x_{n}^2 - a_{n + 1}^2x_{n + 1}^2 = 0`.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import GeneralPythagorean
    >>> from sympy.abc import a, b, c, d, e, x, y, z, t
    >>> GeneralPythagorean(a**2 + b**2 + c**2 - d**2).solve()
    {(t_0**2 + t_1**2 - t_2**2, 2*t_0*t_2, 2*t_1*t_2, t_0**2 + t_1**2 + t_2**2)}
    >>> GeneralPythagorean(9*a**2 - 4*b**2 + 16*c**2 + 25*d**2 + e**2).solve(parameters=[x, y, z, t])
    {(-10*t**2 + 10*x**2 + 10*y**2 + 10*z**2, 15*t**2 + 15*x**2 + 15*y**2 + 15*z**2, 15*t*x, 12*t*y, 60*t*z)}
    """
    name = 'general_pythagorean'

    def matches(self):
        if not (self.total_degree == 2 and self.dimension >= 3):
            return False
        if not self.homogeneous_order:
            return False
        if any((k.is_Mul for k in self.coeff)):
            return False
        if all((self.coeff[k] == 1 for k in self.coeff if k != 1)):
            return False
        if not all((is_square(abs(self.coeff[k])) for k in self.coeff)):
            return False
        return abs(sum((sign(self.coeff[k]) for k in self.coeff))) == self.dimension - 2

    @property
    def n_parameters(self):
        return self.dimension - 1

    def solve(self, parameters=None, limit=1):
        self.pre_solve(parameters)
        coeff = self.coeff
        var = self.free_symbols
        n = self.dimension
        if sign(coeff[var[0] ** 2]) + sign(coeff[var[1] ** 2]) + sign(coeff[var[2] ** 2]) < 0:
            for key in coeff.keys():
                coeff[key] = -coeff[key]
        result = DiophantineSolutionSet(var, parameters=self.parameters)
        index = 0
        for i, v in enumerate(var):
            if sign(coeff[v ** 2]) == -1:
                index = i
        m = result.parameters
        ith = sum((m_i ** 2 for m_i in m))
        L = [ith - 2 * m[n - 2] ** 2]
        L.extend([2 * m[i] * m[n - 2] for i in range(n - 2)])
        sol = L[:index] + [ith] + L[index:]
        lcm = 1
        for i, v in enumerate(var):
            if i == index or (index > 0 and i == 0) or (index == 0 and i == 1):
                lcm = ilcm(lcm, sqrt(abs(coeff[v ** 2])))
            else:
                s = sqrt(coeff[v ** 2])
                lcm = ilcm(lcm, s if _odd(s) else s // 2)
        for i, v in enumerate(var):
            sol[i] = lcm * sol[i] / sqrt(abs(coeff[v ** 2]))
        result.add(sol)
        return result