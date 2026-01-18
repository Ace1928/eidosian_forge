from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
class ComplexCuspCrossSection(CuspCrossSectionBase):
    """
    Similarly to RealCuspCrossSection with the following differences: it
    computes the complex edge lengths and the cusp translations (instead
    of the tilts) and it only works for orientable manifolds.

    The same comment applies about the type of the shapes. The resulting
    edge lengths and translations will be of the same type as the shapes.

    For shapes corresponding to a non-boundary unipotent representation
    (in other words, a manifold having an incomplete cusp), a cusp can
    be developed if an appropriate 1-cocycle is given. The 1-cocycle
    is a cellular cocycle in the dual of the cusp triangulations and
    represents an element in H^1(boundary M; C^*) that must match the
    PSL(2,C) boundary holonomy of the representation.
    It is encoded as dictionary with key (tet index, t3m face, t3m vertex).
    """
    HoroTriangle = ComplexHoroTriangle

    @staticmethod
    def fromManifoldAndShapes(manifold, shapes, one_cocycle=None):
        if not one_cocycle:
            for cusp_info in manifold.cusp_info():
                if not cusp_info['complete?']:
                    raise IncompleteCuspError(manifold)
        if not manifold.is_orientable():
            raise ValueError('Non-orientable')
        m = t3m.Mcomplex(manifold)
        t = TransferKernelStructuresEngine(m, manifold)
        t.reindex_cusps_and_transfer_peripheral_curves()
        t.add_shapes(shapes)
        if one_cocycle == 'develop':
            resolved_one_cocycle = None
        else:
            resolved_one_cocycle = one_cocycle
        c = ComplexCuspCrossSection(m)
        c.add_structures(resolved_one_cocycle)
        c.manifold = manifold
        return c

    def _dummy_for_testing(self):
        """
        Compare the computed edge lengths and tilts against the one computed by
        the SnapPea kernel.

        >>> from snappy import Manifold

        Convention of the kernel is to use (3/8) sqrt(3) as area (ensuring that
        cusp neighborhoods are disjoint).

        >>> cusp_area = 0.649519052838329

        >>> for name in ['m009', 'm015', 't02333']:
        ...     M = Manifold(name)
        ...     e = ComplexCuspCrossSection.fromManifoldAndShapes(M, M.tetrahedra_shapes('rect'))
        ...     e.normalize_cusps(cusp_area)
        ...     e._testing_check_against_snappea(1e-10)

        """

    @staticmethod
    def _get_translation(vertex, ml):
        """
        Compute the translation corresponding to the meridian (ml = 0) or
        longitude (ml = 1) of the given cusp.
        """
        result = 0
        for corner in vertex.Corners:
            tet = corner.Tetrahedron
            subsimplex = corner.Subsimplex
            faces = t3m.simplex.FacesAroundVertexCounterclockwise[subsimplex]
            triangle = tet.horotriangles[subsimplex]
            curves = tet.PeripheralCurves[ml][0][subsimplex]
            for i in range(3):
                this_face = faces[i]
                prev_face = faces[(i + 2) % 3]
                f = curves[this_face] + 2 * curves[prev_face]
                result += f * triangle.lengths[this_face]
        return result / 6

    @staticmethod
    def _compute_translations(vertex):
        vertex.Translations = [ComplexCuspCrossSection._get_translation(vertex, i) for i in range(2)]

    def compute_translations(self):
        for vertex in self.mcomplex.Vertices:
            ComplexCuspCrossSection._compute_translations(vertex)

    @staticmethod
    def _get_normalized_translations(vertex):
        """
        Compute the translations corresponding to the merdian and longitude of
        the given cusp.
        """
        m, l = vertex.Translations
        return (m / l * abs(l), abs(l))

    def all_normalized_translations(self):
        """
        Compute the translations corresponding to the meridian and longitude
        for each cusp.
        """
        self.compute_translations()
        return [ComplexCuspCrossSection._get_normalized_translations(vertex) for vertex in self.mcomplex.Vertices]

    @staticmethod
    def _compute_cusp_shape(vertex):
        m, l = vertex.Translations
        return (l / m).conjugate()

    def cusp_shapes(self):
        """
        Compute the cusp shapes as conjugate of the quotient of the translations
        corresponding to the longitude and meridian for each cusp (SnapPea
        kernel convention).
        """
        self.compute_translations()
        return [ComplexCuspCrossSection._compute_cusp_shape(vertex) for vertex in self.mcomplex.Vertices]

    def add_vertex_positions_to_horotriangles(self):
        """
        Develops cusp to assign to each horotriangle the positions of its three
        vertices in the Euclidean plane.

        Note: For a complete cusp, this is defined only up to translating the
        entire triangle by translations generated by meridian and longitude.

        For an incomplete cusp, this is defined only up to
        similarities generated by the meridian and longitude. The
        positions can be moved such that the fixed point of these
        similarities is at the origin by calling
        move_fixed_point_to_zero after
        add_vertex_positions_to_horotriangles.

        Note: This is not working when one_cocycle is passed during the
        construction of the cusp cross section.
        """
        for cusp in self.mcomplex.Vertices:
            self._add_one_cusp_vertex_positions(cusp)

    def _add_one_cusp_vertex_positions(self, cusp):
        """
        Procedure is similar to _add_one_cusp_cross_section
        """
        corner0 = cusp.Corners[0]
        tet0, vert0 = (corner0.Tetrahedron, corner0.Subsimplex)
        zero = tet0.ShapeParameters[t3m.simplex.E01].parent()(0)
        tet0.horotriangles[vert0].add_vertex_positions(vert0, _pick_an_edge_for_vertex[vert0], zero)
        active = [(tet0, vert0)]
        visited = set()
        while active:
            tet0, vert0 = active.pop()
            for face0 in t3m.simplex.FacesAroundVertexCounterclockwise[vert0]:
                tet1, face1, vert1 = CuspCrossSectionBase._glued_to(tet0, face0, vert0)
                if not (tet1.Index, vert1) in visited:
                    edge0 = _pick_an_edge_for_vertex_and_face[vert0, face0]
                    edge1 = tet0.Gluing[face0].image(edge0)
                    tet1.horotriangles[vert1].add_vertex_positions(vert1, edge1, tet0.horotriangles[vert0].vertex_positions[edge0])
                    active.append((tet1, vert1))
                    visited.add((tet1.Index, vert1))

    def _debug_show_horotriangles(self, cusp=0):
        from sage.all import line, real, imag
        self.add_vertex_positions_to_horotriangles()
        return sum([line([(real(z0), imag(z0)), (real(z1), imag(z1))]) for tet in self.mcomplex.Tetrahedra for V, h in tet.horotriangles.items() for z0 in h.vertex_positions.values() for z1 in h.vertex_positions.values() if tet.Class[V].Index == cusp])

    def _debug_show_lifted_horotriangles(self, cusp=0):
        from sage.all import line, real, imag
        self.add_vertex_positions_to_horotriangles()
        return sum([line([(real(z0), imag(z0)), (real(z1), imag(z1))]) for tet in self.mcomplex.Tetrahedra for V, h in tet.horotriangles.items() for z0 in h.lifted_vertex_positions.values() for z1 in h.lifted_vertex_positions.values() if tet.Class[V].Index == cusp])

    def move_fixed_point_to_zero(self):
        """
        Determines the fixed point of the holonomies for all
        incomplete cusps. Then moves the vertex positions of the
        corresponding cusp triangles so that the fixed point is at the
        origin.

        It also add the boolean v.is_complete to all vertices of the
        triangulation to mark whether the corresponding cusp is
        complete or not.
        """
        for cusp, cusp_info in zip(self.mcomplex.Vertices, self.manifold.cusp_info()):
            cusp.is_complete = cusp_info['complete?']
            if not cusp.is_complete:
                fixed_pt = self._compute_cusp_fixed_point(cusp)
                for corner in cusp.Corners:
                    tet, vert = (corner.Tetrahedron, corner.Subsimplex)
                    trig = tet.horotriangles[vert]
                    trig.vertex_positions = {edge: position - fixed_pt for edge, position in trig.vertex_positions.items()}

    def _compute_cusp_fixed_point(self, cusp):
        """
        Compute fixed point for an incomplete cusp.
        """
        dummy, z, p0, p1 = max(self._compute_cusp_fixed_point_data(cusp), key=lambda d: d[0])
        return (p1 - z * p0) / (1 - z)

    def _compute_cusp_fixed_point_data(self, cusp):
        """
        Compute abs(z-1), z, p0, p1 for each horotriangle, vertex and edge
        as described in _compute_cusp_fixed_point.
        """
        for corner in cusp.Corners:
            tet0, vert0 = (corner.Tetrahedron, corner.Subsimplex)
            vertex_link = _face_edge_face_triples_for_vertex_link[vert0]
            for face0, edge0, other_face in vertex_link:
                tet1, face1, vert1 = CuspCrossSectionBase._glued_to(tet0, face0, vert0)
                edge1 = tet0.Gluing[face0].image(edge0)
                trig0 = tet0.horotriangles[vert0]
                l0 = trig0.lengths[face0]
                p0 = trig0.vertex_positions[edge0]
                trig1 = tet1.horotriangles[vert1]
                l1 = trig1.lengths[face1]
                p1 = trig1.vertex_positions[edge1]
                z = -l1 / l0
                yield (abs(z - 1), z, p0, p1)

    def lift_vertex_positions_of_horotriangles(self):
        """
        After developing an incomplete cusp with
        add_vertex_positions_to_horotriangles, this function moves the
        vertex positions first to zero the fixed point (see
        move_ffixed_point_to_zero) and computes logarithms for all the
        vertex positions of the horotriangles in the Euclidean plane
        in a consistent manner. These logarithms are written to a
        dictionary lifted_vertex_positions on the HoroTriangle's.

        For an incomplete cusp, the respective value in lifted_vertex_positions
        will be None.

        The three logarithms of the vertex positions of a triangle are only
        defined up to adding mu Z + lambda Z where mu and lambda are the
        logarithmic holonomies of the meridian and longitude.
        """
        self.move_fixed_point_to_zero()
        for cusp in self.mcomplex.Vertices:
            self._lift_one_cusp_vertex_positions(cusp)

    def _lift_one_cusp_vertex_positions(self, cusp):
        corner0 = cusp.Corners[0]
        tet0, vert0 = (corner0.Tetrahedron, corner0.Subsimplex)
        trig0 = tet0.horotriangles[vert0]
        edge0 = _pick_an_edge_for_vertex[vert0]
        if cusp.is_complete:
            for corner in cusp.Corners:
                tet0, vert0 = (corner.Tetrahedron, corner.Subsimplex)
                tet0.horotriangles[vert0].lifted_vertex_positions = {vert0 | vert1: None for vert1 in t3m.ZeroSubsimplices if vert0 != vert1}
            return
        trig0.lift_vertex_positions(log(trig0.vertex_positions[edge0]))
        active = [(tet0, vert0)]
        visited = set()
        while active:
            tet0, vert0 = active.pop()
            for face0 in t3m.simplex.FacesAroundVertexCounterclockwise[vert0]:
                tet1, face1, vert1 = CuspCrossSectionBase._glued_to(tet0, face0, vert0)
                if not (tet1.Index, vert1) in visited:
                    edge0 = _pick_an_edge_for_vertex_and_face[vert0, face0]
                    tet1.horotriangles[vert1].lift_vertex_positions(tet0.horotriangles[vert0].lifted_vertex_positions[edge0])
                    active.append((tet1, vert1))
                    visited.add((tet1.Index, vert1))

    def move_lifted_vertex_positions_to_zero_first(self):
        """
        Shift the lifted vertex positions such that the one associated
        to the first vertex when developing the incomplete cusp is
        zero. This makes the values we obtain more stable when
        changing the Dehn-surgery parameters.
        """
        for cusp in self.mcomplex.Vertices:
            if not cusp.is_complete:
                ComplexCuspCrossSection._move_lifted_vertex_positions_cusp(cusp)

    @staticmethod
    def _move_lifted_vertex_positions_cusp(cusp):
        corner0 = cusp.Corners[0]
        tet0, vert0 = (corner0.Tetrahedron, corner0.Subsimplex)
        trig0 = tet0.horotriangles[vert0]
        edge0 = _pick_an_edge_for_vertex[vert0]
        log0 = trig0.lifted_vertex_positions[edge0]
        for corner in cusp.Corners:
            tet, vert = (corner.Tetrahedron, corner.Subsimplex)
            trig = tet.horotriangles[vert]
            trig.lifted_vertex_positions = {edge: position - log0 for edge, position in trig.lifted_vertex_positions.items()}