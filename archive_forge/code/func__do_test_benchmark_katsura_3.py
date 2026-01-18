from sympy.polys.groebnertools import (
from sympy.polys.fglmtools import _representing_matrices
from sympy.polys.orderings import lex, grlex
from sympy.polys.rings import ring, xring
from sympy.polys.domains import ZZ, QQ
from sympy.testing.pytest import slow
from sympy.polys import polyconfig as config
def _do_test_benchmark_katsura_3():
    R, x0, x1, x2 = ring('x:3', ZZ, lex)
    I = [x0 + 2 * x1 + 2 * x2 - 1, x0 ** 2 + 2 * x1 ** 2 + 2 * x2 ** 2 - x0, 2 * x0 * x1 + 2 * x1 * x2 - x1]
    assert groebner(I, R) == [-7 + 7 * x0 + 8 * x2 + 158 * x2 ** 2 - 420 * x2 ** 3, 7 * x1 + 3 * x2 - 79 * x2 ** 2 + 210 * x2 ** 3, x2 + x2 ** 2 - 40 * x2 ** 3 + 84 * x2 ** 4]
    R, x0, x1, x2 = ring('x:3', ZZ, grlex)
    I = [i.set_ring(R) for i in I]
    assert groebner(I, R) == [7 * x1 + 3 * x2 - 79 * x2 ** 2 + 210 * x2 ** 3, -x1 + x2 - 3 * x2 ** 2 + 5 * x1 ** 2, -x1 - 4 * x2 + 10 * x1 * x2 + 12 * x2 ** 2, -1 + x0 + 2 * x1 + 2 * x2]