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
def _complexes_sorted(cls, complexes):
    """Make complex isolating intervals disjoint and sort roots. """
    complexes = cls._refine_complexes(complexes)
    C, F = (0, 1)
    fs = {i[F] for i in complexes}
    for i in range(1, len(complexes)):
        if complexes[i][F] != complexes[i - 1][F]:
            fs.remove(complexes[i - 1][F])
    for i, cmplx in enumerate(complexes):
        assert cmplx[C].conj is (i % 2 == 0)
    cache = {}
    for root, currentfactor, _ in complexes:
        cache.setdefault(currentfactor, []).append(root)
    for currentfactor, root in cache.items():
        _complexes_cache[currentfactor] = root
    return complexes