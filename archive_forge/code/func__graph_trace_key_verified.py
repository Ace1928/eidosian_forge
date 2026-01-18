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
def _graph_trace_key_verified(face_and_signed_distance):
    return face_and_signed_distance[1].center()