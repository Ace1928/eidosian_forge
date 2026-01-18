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
class GeneralSumOfEvenPowers(DiophantineEquationType):
    """
    Representation of the diophantine equation

    `x_{1}^e + x_{2}^e + . . . + x_{n}^e - k = 0`

    where `e` is an even, integer power.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import GeneralSumOfEvenPowers
    >>> from sympy.abc import a, b
    >>> GeneralSumOfEvenPowers(a**4 + b**4 - (2**4 + 3**4)).solve()
    {(2, 3)}

    """
    name = 'general_sum_of_even_powers'

    def matches(self):
        if not self.total_degree > 3:
            return False
        if self.total_degree % 2 != 0:
            return False
        if not all((k.is_Pow and k.exp == self.total_degree for k in self.coeff if k != 1)):
            return False
        return all((self.coeff[k] == 1 for k in self.coeff if k != 1))

    def solve(self, parameters=None, limit=1):
        self.pre_solve(parameters)
        var = self.free_symbols
        coeff = self.coeff
        p = None
        for q in coeff.keys():
            if q.is_Pow and coeff[q]:
                p = q.exp
        k = len(var)
        n = -coeff[1]
        result = DiophantineSolutionSet(var, parameters=self.parameters)
        if n < 0 or limit < 1:
            return result
        sign = [-1 if x.is_nonpositive else 1 for x in var]
        negs = sign.count(-1) != 0
        took = 0
        for t in power_representation(n, p, k):
            if negs:
                result.add([sign[i] * j for i, j in enumerate(t)])
            else:
                result.add(t)
            took += 1
            if took == limit:
                break
        return result