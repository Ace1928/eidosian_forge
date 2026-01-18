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
class GeneralSumOfSquares(DiophantineEquationType):
    """
    Representation of the diophantine equation

    `x_{1}^2 + x_{2}^2 + . . . + x_{n}^2 - k = 0`.

    Details
    =======

    When `n = 3` if `k = 4^a(8m + 7)` for some `a, m \\in Z` then there will be
    no solutions. Refer [1]_ for more details.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import GeneralSumOfSquares
    >>> from sympy.abc import a, b, c, d, e
    >>> GeneralSumOfSquares(a**2 + b**2 + c**2 + d**2 + e**2 - 2345).solve()
    {(15, 22, 22, 24, 24)}

    By default only 1 solution is returned. Use the `limit` keyword for more:

    >>> sorted(GeneralSumOfSquares(a**2 + b**2 + c**2 + d**2 + e**2 - 2345).solve(limit=3))
    [(15, 22, 22, 24, 24), (16, 19, 24, 24, 24), (16, 20, 22, 23, 26)]

    References
    ==========

    .. [1] Representing an integer as a sum of three squares, [online],
        Available:
        https://www.proofwiki.org/wiki/Integer_as_Sum_of_Three_Squares
    """
    name = 'general_sum_of_squares'

    def matches(self):
        if not (self.total_degree == 2 and self.dimension >= 3):
            return False
        if not self.homogeneous_order:
            return False
        if any((k.is_Mul for k in self.coeff)):
            return False
        return all((self.coeff[k] == 1 for k in self.coeff if k != 1))

    def solve(self, parameters=None, limit=1):
        self.pre_solve(parameters)
        var = self.free_symbols
        k = -int(self.coeff[1])
        n = self.dimension
        result = DiophantineSolutionSet(var, parameters=self.parameters)
        if k < 0 or limit < 1:
            return result
        signs = [-1 if x.is_nonpositive else 1 for x in var]
        negs = signs.count(-1) != 0
        took = 0
        for t in sum_of_squares(k, n, zeros=True):
            if negs:
                result.add([signs[i] * j for i, j in enumerate(t)])
            else:
                result.add(t)
            took += 1
            if took == limit:
                break
        return result