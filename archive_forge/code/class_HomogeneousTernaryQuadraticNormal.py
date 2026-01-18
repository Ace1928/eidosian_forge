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
class HomogeneousTernaryQuadraticNormal(DiophantineEquationType):
    """
    Representation of a homogeneous ternary quadratic normal diophantine equation.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.solvers.diophantine.diophantine import HomogeneousTernaryQuadraticNormal
    >>> HomogeneousTernaryQuadraticNormal(4*x**2 - 5*y**2 + z**2).solve()
    {(1, 2, 4)}

    """
    name = 'homogeneous_ternary_quadratic_normal'

    def matches(self):
        if not (self.total_degree == 2 and self.dimension == 3):
            return False
        if not self.homogeneous:
            return False
        if not self.homogeneous_order:
            return False
        nonzero = [k for k in self.coeff if self.coeff[k]]
        return len(nonzero) == 3 and all((i ** 2 in nonzero for i in self.free_symbols))

    def solve(self, parameters=None, limit=None) -> DiophantineSolutionSet:
        self.pre_solve(parameters)
        var = self.free_symbols
        coeff = self.coeff
        x, y, z = var
        a = coeff[x ** 2]
        b = coeff[y ** 2]
        c = coeff[z ** 2]
        (sqf_of_a, sqf_of_b, sqf_of_c), (a_1, b_1, c_1), (a_2, b_2, c_2) = sqf_normal(a, b, c, steps=True)
        A = -a_2 * c_2
        B = -b_2 * c_2
        result = DiophantineSolutionSet(var, parameters=self.parameters)
        if A < 0 and B < 0:
            return result
        if sqrt_mod(-b_2 * c_2, a_2) is None or sqrt_mod(-c_2 * a_2, b_2) is None or sqrt_mod(-a_2 * b_2, c_2) is None:
            return result
        z_0, x_0, y_0 = descent(A, B)
        z_0, q = _rational_pq(z_0, abs(c_2))
        x_0 *= q
        y_0 *= q
        x_0, y_0, z_0 = _remove_gcd(x_0, y_0, z_0)
        if sign(a) == sign(b):
            x_0, y_0, z_0 = holzer(x_0, y_0, z_0, abs(a_2), abs(b_2), abs(c_2))
        elif sign(a) == sign(c):
            x_0, z_0, y_0 = holzer(x_0, z_0, y_0, abs(a_2), abs(c_2), abs(b_2))
        else:
            y_0, z_0, x_0 = holzer(y_0, z_0, x_0, abs(b_2), abs(c_2), abs(a_2))
        x_0 = reconstruct(b_1, c_1, x_0)
        y_0 = reconstruct(a_1, c_1, y_0)
        z_0 = reconstruct(a_1, b_1, z_0)
        sq_lcm = ilcm(sqf_of_a, sqf_of_b, sqf_of_c)
        x_0 = abs(x_0 * sq_lcm // sqf_of_a)
        y_0 = abs(y_0 * sq_lcm // sqf_of_b)
        z_0 = abs(z_0 * sq_lcm // sqf_of_c)
        result.add(_remove_gcd(x_0, y_0, z_0))
        return result