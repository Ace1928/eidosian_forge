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
class GeodesicInfo:
    """
    Information needed to trace a closed geodesic through a triangulation
    given as snappy.snap.t3mlite.Mcomplex with geometric structure added
    by add_r13_geometry.

    The basic information consists of a line in the hyperboloid model
    that is a lift of the closed geodesic and a start and end point on or
    close to that line such that the line segment from the start to the
    end point maps to a simple closed curve in the manifold isotopic to
    the closed geodesic.

    If a client has instantiated this class with the basic information,
    it can call find_tet_or_core_curve. The method find_tet_or_core_curve
    will either:

    1. Detect that the closed geodesic is actually a core curve of a
       filled cusp and set core_curve_cusp and core_curve_direction
       accordingly. This means that instead tracing the geodesic
       through the triangulation, the client has to unfill the
       corresponding cusp instead.
    2. Apply a Decktransformation to the line and points such that
       start point is either in the interior of a tetrahedron (in the
       fundamental domain) or in the union of two (lifted) tetrahedra
       (in the universal cover which is the hyperboloid model). That
       is, if the start point is on a face of the triangulation, it
       will return the two adjacent tetrahedra. If the start point is
       in the interior of a tetrahedron, the client can attempt to
       trace the geodesic through the triangulation. The client can
       use the given (lifted) tetrahedra to develop a tube about the
       geodesic to compute its injectivity radius.

    There is an additional field index that can be used by clients for
    book-keeping purposes, for example, to store the index of the cusp
    obtained by drilling this geodesic.

    The start and end point are unnormalised time-like vectors. Note
    that normalisation is not required for many applications (such as
    computing the intersection of the line segment from the start to
    the end point with a plane) and will enlarge the intervals when
    performing verified computations.
    """

    def __init__(self, mcomplex: Mcomplex, trace: Any, unnormalised_start_point: Any, unnormalised_end_point: Optional[Any]=None, line: Optional[R13LineWithMatrix]=None, tet: Optional[Tetrahedron]=None, lifted_tetrahedra: Sequence[LiftedTetrahedron]=(), core_curve_cusp: Optional[Vertex]=None, core_curve_direction: int=0, index: Optional[int]=None):
        self.mcomplex = mcomplex
        self.trace = trace
        self.unnormalised_start_point = unnormalised_start_point
        self.unnormalised_end_point = unnormalised_end_point
        self.line = line
        self.tet = tet
        self.lifted_tetrahedra = lifted_tetrahedra
        self.core_curve_cusp = core_curve_cusp
        self.core_curve_direction = core_curve_direction
        self.index = index

    def find_tet_or_core_curve(self) -> None:
        """
        Apply Deck-transformations to the start and end point and hyperbolic
        line until we either detected that the given geodesic corresponds to
        a core curve (only if line is not None) or we have captured the start
        point in one or two tetrahedra (in case the start close is on or very
        close to a face).
        This method also computes the distance of the geodesic to the core
        curves (only if line is not None) and raises an exception if we could
        not ensure that this distance is positive.
        """
        self.tet = None
        self.lifted_tetrahedra = ()
        self.core_curve_cusp = None
        self.core_curve_direction = 0
        tet, faces, cusp_curve_vertex = self._graph_trace(self.mcomplex.baseTet)
        if cusp_curve_vertex is not None:
            self.core_curve_direction = self._verify_direction_of_core_curve(tet, cusp_curve_vertex)
            self.core_curve_cusp = tet.Class[cusp_curve_vertex]
            return
        id_matrix = matrix.identity(ring=self.mcomplex.RF, n=4)
        if len(faces) == 0:
            self.tet = tet
            self.lifted_tetrahedra = [LiftedTetrahedron(tet, id_matrix)]
            return
        if len(faces) == 1:
            face, = faces
            other_tet = tet.Neighbor[face]
            other_unnormalised_start_point = tet.O13_matrices[face] * self.unnormalised_start_point
            other_face = tet.Gluing[face].image(face)
            for f in simplex.TwoSubsimplices:
                if f != other_face:
                    if not r13_dot(other_unnormalised_start_point, other_tet.R13_planes[f]) < 0:
                        raise InsufficientPrecisionError('Failed to find lift of geodesic and prove that it intersects tetrahedra of the fundamental domain. Increasing the precision will probably fix this problem.')
            self.lifted_tetrahedra = [LiftedTetrahedron(tet, id_matrix), LiftedTetrahedron(other_tet, other_tet.O13_matrices[other_face])]
            return
        raise InsufficientPrecisionError('Start point chosen on geodesic too close to 1-skeleton of triangulation to verify it is not on the 1-skeleton. Increasing the precision will probably fix this problem.')

    def _graph_trace(self, tet: Tetrahedron) -> Tuple[Tetrahedron, Sequence[int], Optional[int]]:
        """
        Walk from tetrahedron to tetrahedron (transforming start point and
        the other data) to capture the start point in a tetrahedron.
        """
        if self.mcomplex.verified:
            epsilon = 0
            key = _graph_trace_key_verified
        else:
            epsilon = epsilons.compute_epsilon(self.mcomplex.RF)
            key = _graph_trace_key
        entry_cell = simplex.T
        for i in range(constants.graph_trace_max_steps):
            v = self._find_cusp_if_core_curve(tet, entry_cell, epsilon)
            faces_and_signed_distances = [(face, r13_dot(self.unnormalised_start_point, tet.R13_planes[face])) for face in simplex.TwoSubsimplices]
            if v or not any((signed_distance > epsilon for face, signed_distance in faces_and_signed_distances)):
                return (tet, [face for face, signed_distance in faces_and_signed_distances if not signed_distance < -epsilon], v)
            face, worst_distance = max([face_and_signed_distance for face_and_signed_distance in faces_and_signed_distances if face_and_signed_distance[0] != entry_cell], key=key)
            self._transform(tet.O13_matrices[face])
            entry_cell = tet.Gluing[face].image(face)
            tet = tet.Neighbor[face]
        raise exceptions.UnfinishedGraphTraceGeodesicError(constants.graph_trace_max_steps)

    def _transform(self, m):
        """
        Transform the data by matrix.
        """
        self.unnormalised_start_point = m * self.unnormalised_start_point
        if self.unnormalised_end_point:
            self.unnormalised_end_point = m * self.unnormalised_end_point
        if self.line:
            self.line = self.line.transformed(m)

    def _find_cusp_if_core_curve(self, tet: Tetrahedron, entry_cell: int, epsilon) -> Optional[int]:
        """
        Check that the lift of the geodesic is close to the lifts of the core
        curves at the vertices of the tetrahedron adjacent to entry_cell
        where entry_cell is either in simplex.TwoSubsimplices or simplex.T.

        If close, returns the vertex of the tetrahedron (in
        simplex.ZeroSubsimplices), else None.
        """
        if not self.line:
            return None
        for v in simplex.ZeroSubsimplices:
            if not simplex.is_subset(v, entry_cell):
                continue
            core_curve = tet.core_curves.get(v)
            if not core_curve:
                continue
            p = [[r13_dot(pt0, pt1) for pt0 in self.line.r13_line.points] for pt1 in tet.core_curves[v].r13_line.points]
            if not (abs(p[0][0]) > epsilon or abs(p[1][1]) > epsilon):
                return v
            if not (abs(p[0][1]) > epsilon or abs(p[1][0]) > epsilon):
                return v
        return None

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