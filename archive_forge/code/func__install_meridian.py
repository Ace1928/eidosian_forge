from ..snap.t3mlite import Mcomplex
from ..snap.t3mlite import simplex, Tetrahedron
from collections import deque
from typing import Dict
def _install_meridian(start_tet: Tetrahedron) -> None:
    tet = start_tet
    while True:
        for f in [simplex.F2, simplex.F1, simplex.F2, simplex.F3]:
            tet = _walk_face(tet, 0, f)
        if tet is start_tet:
            break