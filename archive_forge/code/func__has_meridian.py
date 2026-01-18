from ..snap.t3mlite import Mcomplex
from ..snap.t3mlite import simplex, Tetrahedron
from collections import deque
from typing import Dict
def _has_meridian(tet: Tetrahedron) -> bool:
    for sheet in tet.PeripheralCurves[0]:
        for v in sheet[simplex.V0].values():
            if v != 0:
                return True
    return False