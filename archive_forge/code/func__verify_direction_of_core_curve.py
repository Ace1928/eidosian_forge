from .line import R13LineWithMatrix
from . import epsilons
from . import constants
from . import exceptions
from ..hyperboloid import r13_dot, o13_inverse, distance_unit_time_r13_points # type: ignore
from ..snap.t3mlite import simplex # type: ignore
from ..snap.t3mlite import Tetrahedron, Vertex, Mcomplex # type: ignore
from ..exceptions import InsufficientPrecisionError # type: ignore
from ..matrix import matrix # type: ignore
from typing import Tuple, Sequence, Optional, Any
def _verify_direction_of_core_curve(self, tet: Tetrahedron, vertex: int) -> int:
    """
        Verify that geodesic and core curve are indeed the same and
        return sign indicating whether they are parallel or anti-parallel.
        """
    if self.line is None:
        raise Exception('There is a bug in the code: it is trying to verify that geodesic is a core curve without being given a line.')
    a = self.line.o13_matrix * self.mcomplex.R13_baseTetInCenter
    m = tet.core_curves[vertex].o13_matrix
    b0 = m * self.mcomplex.R13_baseTetInCenter
    if distance_unit_time_r13_points(a, b0) < self.mcomplex.baseTetInRadius:
        return +1
    b1 = o13_inverse(m) * self.mcomplex.R13_baseTetInCenter
    if distance_unit_time_r13_points(a, b1) < self.mcomplex.baseTetInRadius:
        return -1
    raise InsufficientPrecisionError('Geodesic is very close to a core curve but could not verify it is the core curve. Increasing the precision will probably fix this.')