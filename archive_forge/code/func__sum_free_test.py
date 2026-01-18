from sympy.core import S, Rational
from sympy.combinatorics.schur_number import schur_partition, SchurNumber
from sympy.core.random import _randint
from sympy.testing.pytest import raises
from sympy.core.symbol import symbols
def _sum_free_test(subset):
    """
    Checks if subset is sum-free(There are no x,y,z in the subset such that
    x + y = z)
    """
    for i in subset:
        for j in subset:
            assert (i + j in subset) is False