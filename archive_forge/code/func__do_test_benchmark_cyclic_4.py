from sympy.polys.groebnertools import (
from sympy.polys.fglmtools import _representing_matrices
from sympy.polys.orderings import lex, grlex
from sympy.polys.rings import ring, xring
from sympy.polys.domains import ZZ, QQ
from sympy.testing.pytest import slow
from sympy.polys import polyconfig as config
def _do_test_benchmark_cyclic_4():
    R, a, b, c, d = ring('a,b,c,d', ZZ, lex)
    I = [a + b + c + d, a * b + a * d + b * c + b * d, a * b * c + a * b * d + a * c * d + b * c * d, a * b * c * d - 1]
    assert groebner(I, R) == [4 * a + 3 * d ** 9 - 4 * d ** 5 - 3 * d, 4 * b + 4 * c - 3 * d ** 9 + 4 * d ** 5 + 7 * d, 4 * c ** 2 + 3 * d ** 10 - 4 * d ** 6 - 3 * d ** 2, 4 * c * d ** 4 + 4 * c - d ** 9 + 4 * d ** 5 + 5 * d, d ** 12 - d ** 8 - d ** 4 + 1]
    R, a, b, c, d = ring('a,b,c,d', ZZ, grlex)
    I = [i.set_ring(R) for i in I]
    assert groebner(I, R) == [3 * b * c - c ** 2 + d ** 6 - 3 * d ** 2, -b + 3 * c ** 2 * d ** 3 - c - d ** 5 - 4 * d, -b + 3 * c * d ** 4 + 2 * c + 2 * d ** 5 + 2 * d, c ** 4 + 2 * c ** 2 * d ** 2 - d ** 4 - 2, c ** 3 * d + c * d ** 3 + d ** 4 + 1, b * c ** 2 - c ** 3 - c ** 2 * d - 2 * c * d ** 2 - d ** 3, b ** 2 - c ** 2, b * d + c ** 2 + c * d + d ** 2, a + b + c + d]