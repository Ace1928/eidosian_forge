from ..snap.t3mlite import Mcomplex
from ..snap.t3mlite import simplex, Tetrahedron
from collections import deque
from typing import Dict
def _walk_face(tet: Tetrahedron, ml: int, f: int) -> Tetrahedron:
    """
    Input is a tetrahedron, a number ml saying whether we want to
    set the meridian or longitude and a t3mlite.simplex-style
    face not equal to simplex.F0.

    Add piece to peripheral curve to cusp triangle about vertex 0
    corresponding to walking across the given face. Returns
    tetrahedron after crossing the given face.
    """
    tet.PeripheralCurves[ml][tet.orientation][simplex.V0][f] = +1
    tet = tet.Neighbor[f]
    tet.PeripheralCurves[ml][tet.orientation][simplex.V0][f] = -1
    return tet