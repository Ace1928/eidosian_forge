from ..snap.t3mlite.simplex import *
from ..snap.t3mlite.edge import Edge
from ..snap.t3mlite.arrow import Arrow
from ..snap.t3mlite.tetrahedron import Tetrahedron
from ..snap.t3mlite.mcomplex import VERBOSE
from .exceptions import GeneralPositionError
from .rational_linear_algebra import Vector3, QQ
from . import pl_utils
from . import stored_moves
from .mcomplex_with_expansion import McomplexWithExpansion
from .mcomplex_with_memory import McomplexWithMemory
from .barycentric_geometry import (BarycentricPoint, BarycentricArc,
import random
import collections
import time
def add_arcs_to_standard_solid_tori(mcomplex, num_tori):
    """
    Assumes all the tori are at the end of the list of tets
    """
    M, n = (mcomplex, num_tori)
    assert all((is_standard_solid_torus(tet) for tet in M.Tetrahedra[-n:]))
    for tet in M.Tetrahedra[-n:]:
        add_core_arc_in_one_tet_solid_torus(M, tet)