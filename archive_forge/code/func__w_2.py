from sympy.core import Add, Mul, Symbol, sympify, Dummy, symbols
from sympy.core.containers import Tuple
from sympy.core.singleton import S
from sympy.ntheory import nextprime
from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.domains import ZZ
from sympy.polys.factortools import dup_zz_cyclotomic_poly
from sympy.polys.polyclasses import DMP
from sympy.polys.polytools import Poly, PurePoly
from sympy.polys.polyutils import _analyze_gens
from sympy.utilities import subsets, public, filldedent
from sympy.polys.rings import ring
def _w_2():
    R, x, y = ring('x,y', ZZ)
    return 24 * x ** 8 * y ** 3 + 48 * x ** 8 * y ** 2 + 24 * x ** 7 * y ** 5 - 72 * x ** 7 * y ** 2 + 25 * x ** 6 * y ** 4 + 2 * x ** 6 * y ** 3 + 4 * x ** 6 * y + 8 * x ** 6 + x ** 5 * y ** 6 + x ** 5 * y ** 3 - 12 * x ** 5 + x ** 4 * y ** 5 - x ** 4 * y ** 4 - 2 * x ** 4 * y ** 3 + 292 * x ** 4 * y ** 2 - x ** 3 * y ** 6 + 3 * x ** 3 * y ** 3 - x ** 2 * y ** 5 + 12 * x ** 2 * y ** 3 + 48 * x ** 2 - 12 * y ** 3