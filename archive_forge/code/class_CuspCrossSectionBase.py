from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
class CuspCrossSectionBase(McomplexEngine):
    """
    Base class for RealCuspCrossSection and ComplexCuspCrossSection.
    """

    def add_structures(self, one_cocycle=None):
        self._add_edge_dict()
        self._add_cusp_cross_sections(one_cocycle)

    def _add_edge_dict(self):
        """
        Adds a dictionary that maps a pair of vertices to all edges
        of the triangulation connecting these vertices.
        The key is a pair (v0, v1) of integers with v0 < v1 that are the
        indices of the two vertices.
        """
        self._edge_dict = {}
        for edge in self.mcomplex.Edges:
            vert0, vert1 = edge.Vertices
            key = tuple(sorted([vert0.Index, vert1.Index]))
            self._edge_dict.setdefault(key, []).append(edge)

    def _add_cusp_cross_sections(self, one_cocycle):
        for T in self.mcomplex.Tetrahedra:
            T.horotriangles = {t3m.simplex.V0: None, t3m.simplex.V1: None, t3m.simplex.V2: None, t3m.simplex.V3: None}
        for cusp in self.mcomplex.Vertices:
            self._add_one_cusp_cross_section(cusp, one_cocycle)

    def _add_one_cusp_cross_section(self, cusp, one_cocycle):
        """
        Build a cusp cross section as described in Section 3.6 of the paper

        Asymmetric hyperbolic L-spaces, Heegaard genus, and Dehn filling
        Nathan M. Dunfield, Neil R. Hoffman, Joan E. Licata
        http://arxiv.org/abs/1407.7827
        """
        corner0 = cusp.Corners[0]
        tet0, vert0 = (corner0.Tetrahedron, corner0.Subsimplex)
        face0 = t3m.simplex.FacesAroundVertexCounterclockwise[vert0][0]
        tet0.horotriangles[vert0] = self.HoroTriangle(tet0, vert0, face0, 1)
        active = [(tet0, vert0)]
        while active:
            tet0, vert0 = active.pop()
            for face0 in t3m.simplex.FacesAroundVertexCounterclockwise[vert0]:
                tet1, face1, vert1 = CuspCrossSectionBase._glued_to(tet0, face0, vert0)
                if tet1.horotriangles[vert1] is None:
                    known_side = self.HoroTriangle.direction_sign() * tet0.horotriangles[vert0].lengths[face0]
                    if one_cocycle:
                        known_side *= one_cocycle[tet0.Index, face0, vert0]
                    tet1.horotriangles[vert1] = self.HoroTriangle(tet1, vert1, face1, known_side)
                    active.append((tet1, vert1))

    @staticmethod
    def _glued_to(tetrahedron, face, vertex):
        """
        Returns (other tet, other face, other vertex).
        """
        gluing = tetrahedron.Gluing[face]
        return (tetrahedron.Neighbor[face], gluing.image(face), gluing.image(vertex))

    @staticmethod
    def _cusp_area(cusp):
        area = 0
        for corner in cusp.Corners:
            subsimplex = corner.Subsimplex
            area += corner.Tetrahedron.horotriangles[subsimplex].area
        return area

    def cusp_areas(self):
        """
        List of all cusp areas.
        """
        return [CuspCrossSectionBase._cusp_area(cusp) for cusp in self.mcomplex.Vertices]

    @staticmethod
    def _scale_cusp(cusp, scale):
        for corner in cusp.Corners:
            subsimplex = corner.Subsimplex
            corner.Tetrahedron.horotriangles[subsimplex].rescale(scale)

    def scale_cusps(self, scales):
        """
        Scale each cusp by Euclidean dilation by values in given array.
        """
        for cusp, scale in zip(self.mcomplex.Vertices, scales):
            CuspCrossSectionBase._scale_cusp(cusp, scale)

    def normalize_cusps(self, areas=None):
        """
        Scale cusp so that they have the given target area.
        Without argument, each cusp is scaled to have area 1.
        If the argument is a number, scale each cusp to have that area.
        If the argument is an array, scale each cusp by the respective
        entry in the array.
        """
        current_areas = self.cusp_areas()
        if not areas:
            areas = [1 for area in current_areas]
        elif not isinstance(areas, list):
            areas = [areas for area in current_areas]
        scales = [sqrt(area / current_area) for area, current_area in zip(areas, current_areas)]
        self.scale_cusps(scales)

    def check_cusp_development_exactly(self):
        """
        Check that all side lengths of horo triangles are consistent.
        If the logarithmic edge equations are fulfilled, this implices
        that the all cusps are complete and thus the manifold is complete.
        """
        for tet0 in self.mcomplex.Tetrahedra:
            for vert0 in t3m.simplex.ZeroSubsimplices:
                for face0 in t3m.simplex.FacesAroundVertexCounterclockwise[vert0]:
                    tet1, face1, vert1 = CuspCrossSectionBase._glued_to(tet0, face0, vert0)
                    side0 = tet0.horotriangles[vert0].lengths[face0]
                    side1 = tet1.horotriangles[vert1].lengths[face1]
                    if not side0 == side1 * self.HoroTriangle.direction_sign():
                        raise CuspDevelopmentExactVerifyError(side0, side1)

    @staticmethod
    def _shape_for_edge_embedding(tet, perm):
        """
        Given an edge embedding, find the shape assignment for it.
        If the edge embedding flips orientation, apply conjugate inverse.
        """
        subsimplex = perm.image(3)
        if perm.sign():
            return 1 / tet.ShapeParameters[subsimplex].conjugate()
        else:
            return tet.ShapeParameters[subsimplex]

    def check_polynomial_edge_equations_exactly(self):
        """
        Check that the polynomial edge equations are fulfilled exactly.

        We use the conjugate inverse to support non-orientable manifolds.
        """
        for edge in self.mcomplex.Edges:
            val = 1
            for tet, perm in edge.embeddings():
                val *= CuspCrossSectionBase._shape_for_edge_embedding(tet, perm)
            if not val == 1:
                raise EdgeEquationExactVerifyError(val)

    def check_logarithmic_edge_equations_and_positivity(self, NumericalField):
        """
        Check that the shapes have positive imaginary part and that the
        logarithmic gluing equations have small error.

        The shapes are coerced into the field given as argument before the
        logarithm is computed. It can be, e.g., a ComplexIntervalField.
        """
        for edge in self.mcomplex.Edges:
            log_sum = 0
            for tet, perm in edge.embeddings():
                shape = CuspCrossSectionBase._shape_for_edge_embedding(tet, perm)
                numerical_shape = NumericalField(shape)
                log_shape = log(numerical_shape)
                if not log_shape.imag() > 0:
                    raise ShapePositiveImaginaryPartNumericalVerifyError(numerical_shape)
                log_sum += log_shape
            twoPiI = NumericalField.pi() * NumericalField(2j)
            if not abs(log_sum - twoPiI) < NumericalField(1e-07):
                raise EdgeEquationLogLiftNumericalVerifyError(log_sum)

    def _testing_check_against_snappea(self, epsilon):
        ZeroSubs = t3m.simplex.ZeroSubsimplices
        snappea_tilts, snappea_edges = self.manifold._cusp_cross_section_info()
        for tet, snappea_tet_edges in zip(self.mcomplex.Tetrahedra, snappea_edges):
            for v, snappea_triangle_edges in zip(ZeroSubs, snappea_tet_edges):
                for f, snappea_triangle_edge in zip(ZeroSubs, snappea_triangle_edges):
                    if v != f:
                        F = t3m.simplex.comp(f)
                        length = abs(tet.horotriangles[v].lengths[F])
                        if not abs(length - snappea_triangle_edge) < epsilon:
                            raise ConsistencyWithSnapPeaNumericalVerifyError(snappea_triangle_edge, length)

    @staticmethod
    def _lower_bound_max_area_triangle_for_std_form(z):
        """
        Imagine an ideal tetrahedron in the upper half space model with
        vertices at 0, 1, z, and infinity. Pick the lowest (horizontal)
        horosphere about infinity that intersects the tetrahedron in a
        triangle, i.e, just touches the face opposite to infinity.
        This method will return the hyperbolic area of that triangle.

        The result is the same for z, 1/(1-z), and 1 - 1/z.
        """
        if z.real() < 0:
            return 2 * z.imag() / abs(z - 1) ** 2
        if z.real() > 1:
            return 2 * z.imag() / abs(z) ** 2
        if abs(2 * z - 1) < 1:
            return 2 * z.imag()
        return 2 * z.imag() ** 3 / (abs(z) * abs(z - 1)) ** 2

    def ensure_std_form(self, allow_scaling_up=False):
        """
        Makes sure that the cusp neighborhoods intersect each tetrahedron
        in standard form by scaling the cusp neighborhoods down if necessary.
        """
        z = self.mcomplex.Tetrahedra[0].ShapeParameters[t3m.simplex.E01]
        RF = z.real().parent()
        if allow_scaling_up:
            area_scales = [[] for v in self.mcomplex.Vertices]
        else:
            area_scales = [[RF(1)] for v in self.mcomplex.Vertices]
        for tet in self.mcomplex.Tetrahedra:
            z = tet.ShapeParameters[t3m.simplex.E01]
            max_area = ComplexCuspCrossSection._lower_bound_max_area_triangle_for_std_form(z)
            for zeroSubsimplex, triangle in tet.horotriangles.items():
                area_scale = max_area / triangle.area
                vertex = tet.Class[zeroSubsimplex]
                area_scales[vertex.Index].append(area_scale)
        scales = [sqrt(correct_min(s)) for s in area_scales]
        self.scale_cusps(scales)

    @staticmethod
    def _exp_distance_edge(edge):
        """
        Given an edge, returns the exp of the (hyperbolic) distance of the
        two cusp neighborhoods at the ends of the edge measured along that
        edge.
        """
        tet, perm = next(edge.embeddings())
        face = 15 - (1 << perm[3])
        ptolemy_sqr = tet.horotriangles[1 << perm[0]].lengths[face] * tet.horotriangles[1 << perm[1]].lengths[face]
        return abs(1 / ptolemy_sqr)

    @staticmethod
    def _exp_distance_of_edges(edges):
        """
        Given edges between two (not necessarily distinct) cusps,
        compute the exp of the smallest (hyperbolic) distance of the
        two cusp neighborhoods measured along all the given edges.
        """
        return correct_min([ComplexCuspCrossSection._exp_distance_edge(edge) for edge in edges])

    def ensure_disjoint_on_edges(self):
        """
        Scales the cusp neighborhoods down until they are disjoint when
        intersected with the edges of the triangulations.

        Given an edge of a triangulation, we can easily compute the signed
        distance between the two cusp neighborhoods at the ends of the edge
        measured along that edge. Thus, we can easily check that all the
        distances measured along all the edges are positive and scale the
        cusps down if necessary.

        Unfortunately, this is not sufficient to ensure that two cusp
        neighborhoods are disjoint since there might be a geodesic between
        the two cusps such that the distance between the two cusps measured
        along the geodesic is shorter than measured along any edge of the
        triangulation.

        Thus, it is necessary to call ensure_std_form as well:
        it will make sure that the cusp neighborhoods are small enough so
        that they intersect the tetrahedra in "standard" form.
        Here, "standard" form means that the corresponding horoball about a
        vertex of a tetrahedron intersects the three faces of the tetrahedron
        adjacent to the vertex but not the one opposite to the vertex.

        For any geometric triangulation, standard form and positive distance
        measured along all edges of the triangulation is sufficient for
        disjoint neighborhoods.

        The SnapPea kernel uses the proto-canonical triangulation associated
        to the cusp neighborhood to get around this when computing the
        "reach" and the "stoppers" for the cusps.

        **Remark:** This means that the cusp neighborhoods might be scaled down
        more than necessary. Related open questions are: given maximal disjoint
        cusp neighborhoods (maximal in the sense that no neighborhood can be
        expanded without bumping into another or itself), is there always a
        geometric triangulation intersecting the cusp neighborhoods in standard
        form? Is there an easy algorithm to find this triangulation, e.g., by
        applying a 2-3 move whenever we see a non-standard intersection?
        """
        num_cusps = len(self.mcomplex.Vertices)
        for i in range(num_cusps):
            if (i, i) in self._edge_dict:
                dist = ComplexCuspCrossSection._exp_distance_of_edges(self._edge_dict[i, i])
                if not dist > 1:
                    scale = sqrt(dist)
                    ComplexCuspCrossSection._scale_cusp(self.mcomplex.Vertices[i], scale)
        for i in range(num_cusps):
            for j in range(i):
                if (j, i) in self._edge_dict:
                    dist = ComplexCuspCrossSection._exp_distance_of_edges(self._edge_dict[j, i])
                    if not dist > 1:
                        scale = sqrt(dist)
                        ComplexCuspCrossSection._scale_cusp(self.mcomplex.Vertices[i], scale)
                        ComplexCuspCrossSection._scale_cusp(self.mcomplex.Vertices[j], scale)