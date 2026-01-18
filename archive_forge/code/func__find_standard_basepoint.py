from .line import R13LineWithMatrix
from ..verify.shapes import compute_hyperbolic_shapes # type: ignore
from ..snap.fundamental_polyhedron import FundamentalPolyhedronEngine # type: ignore
from ..snap.kernel_structures import TransferKernelStructuresEngine # type: ignore
from ..snap.t3mlite import simplex, Mcomplex, Tetrahedron, Vertex # type: ignore
from ..SnapPy import word_as_list # type: ignore
from ..hyperboloid import (o13_inverse,  # type: ignore
from ..upper_halfspace import sl2c_inverse, psl2c_to_o13 # type: ignore
from ..upper_halfspace.ideal_point import ideal_point_to_r13 # type: ignore
from ..matrix import vector, matrix, mat_solve # type: ignore
from ..math_basics import prod, xgcd # type: ignore
from collections import deque
from typing import Tuple, Sequence, Optional, Any
def _find_standard_basepoint(mcomplex: Mcomplex, vertex: Vertex) -> Tuple[Tetrahedron, int]:
    """
    Reimplements find_standard_basepoint in fundamental_group.c.

    That is, it finds the same tetrahedron and vertex of that tetrahedron
    in the fundamental domain that the SnapPea kernel used to compute the
    words for the meridian and longitude of the given cusp.

    The SnapPea kernel picks the first vertex it finds where the meridian
    and longitude intersect.
    """
    for tet in mcomplex.Tetrahedra:
        for v in simplex.ZeroSubsimplices:
            if tet.Class[v] is vertex:
                for f in simplex.TwoSubsimplices:
                    if tet.PeripheralCurves[0][0][v][f] != 0 and tet.PeripheralCurves[1][0][v][f] != 0:
                        return (tet, v)
    raise Exception('Could not find basepoint for cusp. This is a bug.')