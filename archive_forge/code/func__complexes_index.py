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
def _complexes_index(cls, complexes, index):
    """
        Map initial complex root index to an index in a factor where
        the root belongs.
        """
    i = 0
    for j, (_, currentfactor, k) in enumerate(complexes):
        if index < i + k:
            poly, index = (currentfactor, 0)
            for _, currentfactor, _ in complexes[:j]:
                if currentfactor == poly:
                    index += 1
            index += len(_reals_cache[poly])
            return (poly, index)
        else:
            i += k