from sympy.polys.groebnertools import (
from sympy.polys.fglmtools import _representing_matrices
from sympy.polys.orderings import lex, grlex
from sympy.polys.rings import ring, xring
from sympy.polys.domains import ZZ, QQ
from sympy.testing.pytest import slow
from sympy.polys import polyconfig as config
def _do_test_benchmark_minpoly():
    R, x, y, z = ring('x,y,z', QQ, lex)
    F = [x ** 3 + x + 1, y ** 2 + y + 1, (x + y) * z - (x ** 2 + y)]
    G = [x + QQ(155, 2067) * z ** 5 - QQ(355, 689) * z ** 4 + QQ(6062, 2067) * z ** 3 - QQ(3687, 689) * z ** 2 + QQ(6878, 2067) * z - QQ(25, 53), y + QQ(4, 53) * z ** 5 - QQ(91, 159) * z ** 4 + QQ(523, 159) * z ** 3 - QQ(387, 53) * z ** 2 + QQ(1043, 159) * z - QQ(308, 159), z ** 6 - 7 * z ** 5 + 41 * z ** 4 - 82 * z ** 3 + 89 * z ** 2 - 46 * z + 13]
    assert groebner(F, R) == G