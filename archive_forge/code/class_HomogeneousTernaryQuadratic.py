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
class HomogeneousTernaryQuadratic(DiophantineEquationType):
    """
    Representation of a homogeneous ternary quadratic diophantine equation.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.solvers.diophantine.diophantine import HomogeneousTernaryQuadratic
    >>> HomogeneousTernaryQuadratic(x**2 + y**2 - 3*z**2 + x*y).solve()
    {(-1, 2, 1)}
    >>> HomogeneousTernaryQuadratic(3*x**2 + y**2 - 3*z**2 + 5*x*y + y*z).solve()
    {(3, 12, 13)}

    """
    name = 'homogeneous_ternary_quadratic'

    def matches(self):
        if not (self.total_degree == 2 and self.dimension == 3):
            return False
        if not self.homogeneous:
            return False
        if not self.homogeneous_order:
            return False
        nonzero = [k for k in self.coeff if self.coeff[k]]
        return not (len(nonzero) == 3 and all((i ** 2 in nonzero for i in self.free_symbols)))

    def solve(self, parameters=None, limit=None):
        self.pre_solve(parameters)
        _var = self.free_symbols
        coeff = self.coeff
        x, y, z = _var
        var = [x, y, z]
        result = DiophantineSolutionSet(var, parameters=self.parameters)

        def unpack_sol(sol):
            if len(sol) > 0:
                return list(sol)[0]
            return (None, None, None)
        if not any((coeff[i ** 2] for i in var)):
            if coeff[x * z]:
                sols = diophantine(coeff[x * y] * x + coeff[y * z] * z - x * z)
                s = sols.pop()
                min_sum = abs(s[0]) + abs(s[1])
                for r in sols:
                    m = abs(r[0]) + abs(r[1])
                    if m < min_sum:
                        s = r
                        min_sum = m
                result.add(_remove_gcd(s[0], -coeff[x * z], s[1]))
                return result
            else:
                var[0], var[1] = (_var[1], _var[0])
                y_0, x_0, z_0 = unpack_sol(_diop_ternary_quadratic(var, coeff))
                if x_0 is not None:
                    result.add((x_0, y_0, z_0))
                return result
        if coeff[x ** 2] == 0:
            if coeff[y ** 2] == 0:
                var[0], var[2] = (_var[2], _var[0])
                z_0, y_0, x_0 = unpack_sol(_diop_ternary_quadratic(var, coeff))
            else:
                var[0], var[1] = (_var[1], _var[0])
                y_0, x_0, z_0 = unpack_sol(_diop_ternary_quadratic(var, coeff))
        elif coeff[x * y] or coeff[x * z]:
            A = coeff[x ** 2]
            B = coeff[x * y]
            C = coeff[x * z]
            D = coeff[y ** 2]
            E = coeff[y * z]
            F = coeff[z ** 2]
            _coeff = {}
            _coeff[x ** 2] = 4 * A ** 2
            _coeff[y ** 2] = 4 * A * D - B ** 2
            _coeff[z ** 2] = 4 * A * F - C ** 2
            _coeff[y * z] = 4 * A * E - 2 * B * C
            _coeff[x * y] = 0
            _coeff[x * z] = 0
            x_0, y_0, z_0 = unpack_sol(_diop_ternary_quadratic(var, _coeff))
            if x_0 is None:
                return result
            p, q = _rational_pq(B * y_0 + C * z_0, 2 * A)
            x_0, y_0, z_0 = (x_0 * q - p, y_0 * q, z_0 * q)
        elif coeff[z * y] != 0:
            if coeff[y ** 2] == 0:
                if coeff[z ** 2] == 0:
                    A = coeff[x ** 2]
                    E = coeff[y * z]
                    b, a = _rational_pq(-E, A)
                    x_0, y_0, z_0 = (b, a, b)
                else:
                    var[0], var[2] = (_var[2], _var[0])
                    z_0, y_0, x_0 = unpack_sol(_diop_ternary_quadratic(var, coeff))
            else:
                var[0], var[1] = (_var[1], _var[0])
                y_0, x_0, z_0 = unpack_sol(_diop_ternary_quadratic(var, coeff))
        else:
            x_0, y_0, z_0 = unpack_sol(_diop_ternary_quadratic_normal(var, coeff))
        if x_0 is None:
            return result
        result.add(_remove_gcd(x_0, y_0, z_0))
        return result