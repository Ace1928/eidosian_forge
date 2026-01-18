from .geodesic_info import GeodesicInfo
from .line import R13LineWithMatrix, distance_r13_lines
from . import constants
from . import epsilons
from . import exceptions
from ..snap.t3mlite import simplex, Tetrahedron, Mcomplex # type: ignore
from ..hyperboloid import r13_dot # type: ignore
from ..exceptions import InsufficientPrecisionError # type: ignore
from typing import Sequence, Optional, List
def _verify_away_from_core_curve(line: Optional[R13LineWithMatrix], tet: Tetrahedron, face: int, epsilon):
    """
    If the geodesic is intersecting a core curve, the tracing would
    fail in that it would never reach the intersection point and thus
    either hit the iteration limit or breaking down because of
    rounding-errors.

    This function is catching this case to give a meaningful exception
    faster. It does so by computing the distance between the lift of
    the geodesic we are tracing and the lifts of the core curve
    corresponding to the vertices of the tetrahedra adjacent to the
    given face.
    """
    if line is None:
        return
    for v in simplex.ZeroSubsimplices:
        if not simplex.is_subset(v, face):
            continue
        core_curve: Optional[R13LineWithMatrix] = tet.core_curves.get(v)
        if core_curve is None:
            continue
        d = distance_r13_lines(core_curve.r13_line, line.r13_line)
        if not d > epsilon:
            raise exceptions.GeodesicCloseToCoreCurve()