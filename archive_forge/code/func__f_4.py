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
def _f_4():
    R, x, y, z = ring('x,y,z', ZZ)
    return -x ** 9 * y ** 8 * z - x ** 8 * y ** 5 * z ** 3 - x ** 7 * y ** 12 * z ** 2 - 5 * x ** 7 * y ** 8 - x ** 6 * y ** 9 * z ** 4 + x ** 6 * y ** 7 * z ** 3 + 3 * x ** 6 * y ** 7 * z - 5 * x ** 6 * y ** 5 * z ** 2 - x ** 6 * y ** 4 * z ** 3 + x ** 5 * y ** 4 * z ** 5 + 3 * x ** 5 * y ** 4 * z ** 3 - x ** 5 * y * z ** 5 + x ** 4 * y ** 11 * z ** 4 + 3 * x ** 4 * y ** 11 * z ** 2 - x ** 4 * y ** 8 * z ** 4 + 5 * x ** 4 * y ** 7 * z ** 2 + 15 * x ** 4 * y ** 7 - 5 * x ** 4 * y ** 4 * z ** 2 + x ** 3 * y ** 8 * z ** 6 + 3 * x ** 3 * y ** 8 * z ** 4 - x ** 3 * y ** 5 * z ** 6 + 5 * x ** 3 * y ** 4 * z ** 4 + 15 * x ** 3 * y ** 4 * z ** 2 + x ** 3 * y ** 3 * z ** 5 + 3 * x ** 3 * y ** 3 * z ** 3 - 5 * x ** 3 * y * z ** 4 + x ** 2 * z ** 7 + 3 * x ** 2 * z ** 5 + x * y ** 7 * z ** 6 + 3 * x * y ** 7 * z ** 4 + 5 * x * y ** 3 * z ** 4 + 15 * x * y ** 3 * z ** 2 + y ** 4 * z ** 8 + 3 * y ** 4 * z ** 6 + 5 * z ** 6 + 15 * z ** 4