from . import constants
from . import exceptions
from . import epsilons
from .line import distance_r13_lines, R13Line, R13LineWithMatrix
from .geodesic_info import GeodesicInfo, LiftedTetrahedron
from .quotient_space import balance_end_points_of_line, ZQuotientLiftedTetrahedronSet
from ..hyperboloid import ( # type: ignore
from ..snap.t3mlite import simplex, Tetrahedron, Mcomplex # type: ignore
from ..matrix import matrix # type: ignore
from ..math_basics import is_RealIntervalFieldElement # type: ignore
from ..exceptions import InsufficientPrecisionError # type: ignore
import heapq
from typing import Sequence, Any
class GeodesicTubePiece:
    """
    A class for the pieces produced by GeodesicTube to cover a tube T about
    the given geodesic of the given radius r in the given manifold.

    Such a piece is encoded as a tetrahedron and a line L in the
    hyperboloiod model. Imagine that tetrahedron as part of the
    fundamental domain and intersect it with a tube about L with
    radius r. The images of these intersections in the manifold
    cover T. Because we error on the side of rather adding than dropping
    a piece when using interval arithmetic, the union of the images might not
    be exactly T but a superset. A piece also stores a lower bound for the
    distance between its tetrahedron in the fundamental domain and L.

    In other words, GeodesicTube produces a piece for each tetrahedron
    in the fundamental domain and each lift of the closed geodesic
    to the hyperboloid model for which the above distance is less than r
    (or more accurately, could not be proven to be greater than r).

    When using verified computation, lower_bound is an interval for
    convenience, even though only the left value of the interval is
    relevant.
    """

    def __init__(self, tet: Tetrahedron, lifted_geodesic: R13Line, lower_bound):
        self.tet = tet
        self.lifted_geodesic = lifted_geodesic
        self.lower_bound = lower_bound