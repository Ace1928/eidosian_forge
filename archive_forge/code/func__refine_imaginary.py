from sympy.core.basic import Basic
from sympy.core import (S, Expr, Integer, Float, I, oo, Add, Lambda,
from sympy.core.cache import cacheit
from sympy.core.relational import is_le
from sympy.core.sorting import ordered
from sympy.polys.domains import QQ
from sympy.polys.polyerrors import (
from sympy.polys.polyfuncs import symmetrize, viete
from sympy.polys.polyroots import (
from sympy.polys.polytools import Poly, PurePoly, factor
from sympy.polys.rationaltools import together
from sympy.polys.rootisolation import (
from sympy.utilities import lambdify, public, sift, numbered_symbols
from mpmath import mpf, mpc, findroot, workprec
from mpmath.libmp.libmpf import dps_to_prec, prec_to_dps
from sympy.multipledispatch import dispatch
from itertools import chain
@classmethod
def _refine_imaginary(cls, complexes):
    sifted = sift(complexes, lambda c: c[1])
    complexes = []
    for f in ordered(sifted):
        nimag = _imag_count_of_factor(f)
        if nimag == 0:
            for u, f, k in sifted[f]:
                while u.ax * u.bx <= 0:
                    u = u._inner_refine()
                complexes.append((u, f, k))
        else:
            potential_imag = list(range(len(sifted[f])))
            while True:
                assert len(potential_imag) > 1
                for i in list(potential_imag):
                    u, f, k = sifted[f][i]
                    if u.ax * u.bx > 0:
                        potential_imag.remove(i)
                    elif u.ax != u.bx:
                        u = u._inner_refine()
                        sifted[f][i] = (u, f, k)
                if len(potential_imag) == nimag:
                    break
            complexes.extend(sifted[f])
    return complexes