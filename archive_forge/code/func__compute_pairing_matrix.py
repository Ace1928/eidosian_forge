from .mcomplex_base import *
from .kernel_structures import *
from . import t3mlite as t3m
from .t3mlite import ZeroSubsimplices, simplex
from .t3mlite import Corner, Perm4
from .t3mlite import V0, V1, V2, V3
from ..math_basics import prod
from functools import reduce
from ..sage_helper import _within_sage
def _compute_pairing_matrix(pairing):
    (inCorner, outCorner), perm = pairing
    inTriple = []
    outTriple = []
    for v in simplex.ZeroSubsimplices:
        if simplex.is_subset(v, inCorner.Subsimplex):
            inTriple.append(inCorner.Tetrahedron.Class[v].IdealPoint)
            outTriple.append(outCorner.Tetrahedron.Class[perm.image(v)].IdealPoint)
    return _matrix_taking_triple_to_triple(outTriple, inTriple)