from snappy.snap import t3mlite as t3m
from snappy import Triangulation
from snappy.SnapPy import matrix, vector
from snappy.snap.mcomplex_base import *
from snappy.verify.cuspCrossSection import *
from ..upper_halfspace import pgl2c_to_o13, sl2c_inverse
from ..upper_halfspace.ideal_point import ideal_point_to_r13
from .hyperboloid_utilities import *
from .upper_halfspace_utilities import *
from .raytracing_data import *
from math import sqrt
class IdealRaytracingData(RaytracingData):
    """
    Given a SnapPy manifold, computes data for the shader fragment.glsl
    to raytrace the inside view::

        >>> from snappy import *
        >>> data = IdealRaytracingData.from_manifold(Manifold("m004"))
        >>> data = IdealRaytracingData.from_manifold(ManifoldHP("m004"))

    The values that need to be pushed into the shader's uniforms can
    be obtained as dictionary::

        >>> data.get_uniform_bindings() # doctest: +ELLIPSIS
        {...}

    The compile time constants can similarly be obtained as dictionary::

        >>> data.get_compile_time_constants() # doctest: +ELLIPSIS
        {...}

    The shader needs to know in what tetrahedron and where in the tetrahedron
    the camera is. This is encoded as pair matrix and tetrahedron index::

        >>> view_state = (matrix([[ 1.0, 0.0, 0.0, 0.0],
        ...                       [ 0.0, 1.0, 0.0, 0.0],
        ...                       [ 0.0, 0.0, 0.0,-1.0],
        ...                       [ 0.0, 0.0, 1.0, 0.0]]), 0, 0.0)

    To move/rotate the camera which might potentially put the camera
    into a different tetrahedron, the new pair can be computed as
    follows::

        >>> m = matrix([[ 3.0 , 0.0 , 2.82, 0.0 ],
        ...             [ 0.0 , 1.0 , 0.0 , 0.0 ],
        ...             [ 2.82, 0.0 , 3.0 , 0.0 ],
        ...             [ 0.0 , 0.0 , 0.0 , 1.0 ]])
        >>> view_state = data.update_view_state(view_state, m)
        >>> view_state    # doctest: +NUMERIC6
        ([     1.08997684        1e-16   0.43364676        1e-16 ]
        [          1e-16  -1.00000000         1e-16       1e-16 ]
        [    -0.43364676        1e-16  -1.08997684        1e-16 ]
        [          1e-16        1e-16        1e-16   1.00000000 ], 1, 0.0)

    """

    @staticmethod
    def from_manifold(manifold, areas=None, insphere_scale=0.05, weights=None):
        if manifold.solution_type() != 'all tetrahedra positively oriented':
            return NonGeometricRaytracingData(t3m.Mcomplex(manifold))
        num_cusps = manifold.num_cusps()
        snappy_trig = Triangulation(manifold)
        snappy_trig.dehn_fill(num_cusps * [(0, 0)])
        c = ComplexCuspCrossSection.fromManifoldAndShapes(manifold, manifold.tetrahedra_shapes('rect'), one_cocycle='develop')
        c.normalize_cusps()
        c.compute_translations()
        c.add_vertex_positions_to_horotriangles()
        c.lift_vertex_positions_of_horotriangles()
        c.move_lifted_vertex_positions_to_zero_first()
        r = IdealRaytracingData(c.mcomplex, manifold)
        z = c.mcomplex.Tetrahedra[0].ShapeParameters[t3m.E01]
        r.RF = z.real().parent()
        r.insphere_scale = r.RF(insphere_scale)
        resolved_areas = num_cusps * [1.0] if areas is None else areas
        r.areas = [r.RF(area) for area in resolved_areas]
        r.peripheral_gluing_equations = snappy_trig.gluing_equations()[snappy_trig.num_tetrahedra():]
        r._add_complex_vertices()
        r._add_R13_vertices()
        r._add_O13_matrices_to_faces()
        r._add_R13_planes_to_faces()
        r._add_R13_horosphere_scales_to_vertices()
        r._add_cusp_to_tet_matrices()
        r._add_margulis_tube_ends()
        r._add_inspheres()
        r._add_log_holonomies()
        r._add_cusp_triangle_vertex_positions()
        r.add_weights(weights)
        return r

    def __init__(self, mcomplex, snappy_manifold):
        super(IdealRaytracingData, self).__init__(mcomplex)
        self.snappy_manifold = snappy_manifold

    def _add_O13_matrices_to_faces(self):
        for tet in self.mcomplex.Tetrahedra:
            tet.O13_matrices = {F: _o13_matrix_for_face(tet, F) for F in t3m.TwoSubsimplices}

    def _add_complex_vertices(self):
        for tet in self.mcomplex.Tetrahedra:
            tet.complex_vertices = {v: vert for v, vert in zip(t3m.ZeroSubsimplices, symmetric_vertices_for_tetrahedron(tet.ShapeParameters[t3m.E01]))}

    def _add_R13_vertices(self):
        for tet in self.mcomplex.Tetrahedra:
            tet.R13_vertices = {V: ideal_point_to_r13(z, self.RF) for V, z in tet.complex_vertices.items()}
            tet.R13_vertex_products = {v0 | v1: r13_dot(pt0, pt1) for v0, pt0 in tet.R13_vertices.items() for v1, pt1 in tet.R13_vertices.items() if v0 != v1}

    def _add_R13_planes_to_faces(self):
        for tet in self.mcomplex.Tetrahedra:
            planes = make_tet_planes([tet.R13_vertices[v] for v in t3m.ZeroSubsimplices])
            tet.R13_planes = {F: plane for F, plane in zip(t3m.TwoSubsimplices, planes)}

    def _compute_R13_horosphere_scale_for_vertex(self, tet, V0):
        vertex = tet.Class[V0]
        if not vertex.is_complete:
            return 0.0
        area = self.areas[vertex.Index]
        if area < 1e-06:
            return 0.0
        V1, V2, _ = t3m.VerticesOfFaceCounterclockwise[t3m.comp(V0)]
        cusp_length = tet.horotriangles[V0].get_real_lengths()[V0 | V1 | V2]
        scale_for_unit_length = (-2 * tet.R13_vertex_products[V1 | V2] / (tet.R13_vertex_products[V0 | V1] * tet.R13_vertex_products[V0 | V2])).sqrt()
        return scale_for_unit_length / (cusp_length * area.sqrt())

    def _add_R13_horosphere_scales_to_vertices(self):
        for tet in self.mcomplex.Tetrahedra:
            tet.R13_horosphere_scales = {V: self._compute_R13_horosphere_scale_for_vertex(tet, V) for V in t3m.ZeroSubsimplices}

    def _add_cusp_triangle_vertex_positions(self):
        for tet in self.mcomplex.Tetrahedra:
            tet.cusp_triangle_vertex_positions = {V: _compute_cusp_triangle_vertex_positions(tet, V, i) for i, V in enumerate(t3m.ZeroSubsimplices)}

    def _add_cusp_to_tet_matrices(self):
        for tet in self.mcomplex.Tetrahedra:
            m = [(V, _compute_cusp_to_tet_and_inverse_matrices(tet, V, i)) for i, V in enumerate(t3m.ZeroSubsimplices)]
            tet.cusp_to_tet_matrices = {V: m1 for V, (m1, m2) in m}
            tet.tet_to_cusp_matrices = {V: m2 for V, (m1, m2) in m}

    def _add_margulis_tube_ends(self):
        for tet in self.mcomplex.Tetrahedra:
            tet.margulisTubeEnds = {vertex: _compute_margulis_tube_ends(tet, vertex) for vertex in t3m.ZeroSubsimplices}

    def _add_inspheres(self):
        for tet in self.mcomplex.Tetrahedra:
            tet.inradius = tet.R13_planes[t3m.F0][0].arcsinh()
            tmp = tet.inradius * self.insphere_scale
            tet.cosh_sqr_inradius = tmp.cosh() ** 2

    def _add_log_holonomies_to_cusp(self, cusp, shapes):
        i = cusp.Index
        if cusp.is_complete:
            m_param, l_param = cusp.Translations
        else:
            m_param, l_param = [sum((shape * expo for shape, expo in zip(shapes, self.peripheral_gluing_equations[2 * i + j]))) for j in range(2)]
        a, c = (m_param.real(), m_param.imag())
        b, d = (l_param.real(), l_param.imag())
        det = a * d - b * c
        cusp.mat_log = matrix([[d, -b], [-c, a]]) / det
        if cusp.is_complete:
            cusp.margulisTubeRadiusParam = 0.0
        else:
            slope = 2 * self.areas[i] / abs(det)
            x = (slope ** 2 / (slope ** 2 + 1)).sqrt()
            y = (1 / (slope ** 2 + 1)).sqrt()
            rSqr = 1 + (x ** 2 + (1 - y) ** 2) / (2 * y)
            cusp.margulisTubeRadiusParam = 0.25 * (1.0 + rSqr)

    def _add_log_holonomies(self):
        shapes = [tet.ShapeParameters[e].log() for tet in self.mcomplex.Tetrahedra for e in [t3m.E01, t3m.E02, t3m.E03]]
        for cusp, cusp_info in zip(self.mcomplex.Vertices, self.snappy_manifold.cusp_info()):
            self._add_log_holonomies_to_cusp(cusp, shapes)

    def get_uniform_bindings(self):
        d = super(IdealRaytracingData, self).get_uniform_bindings()
        orientations = [+1 if tet.ShapeParameters[t3m.E01].imag() > 0 else -1 for tet in self.mcomplex.Tetrahedra]
        horosphere_scales = [tet.R13_horosphere_scales[V] for tet in self.mcomplex.Tetrahedra for V in t3m.ZeroSubsimplices]
        margulisTubeTails = [tet.margulisTubeEnds[V][0] for tet in self.mcomplex.Tetrahedra for V in t3m.ZeroSubsimplices]
        margulisTubeHeads = [tet.margulisTubeEnds[V][1] for tet in self.mcomplex.Tetrahedra for V in t3m.ZeroSubsimplices]
        margulisTubeRadiusParams = [tet.Class[V].margulisTubeRadiusParam for tet in self.mcomplex.Tetrahedra for V in t3m.ZeroSubsimplices]
        cusp_to_tet_matrices = [tet.cusp_to_tet_matrices[V] for tet in self.mcomplex.Tetrahedra for V in t3m.ZeroSubsimplices]
        tet_to_cusp_matrices = [tet.tet_to_cusp_matrices[V] for tet in self.mcomplex.Tetrahedra for V in t3m.ZeroSubsimplices]
        cusp_translations = [[[z.real(), z.imag()] for z in tet.Class[V].Translations] for tet in self.mcomplex.Tetrahedra for V in t3m.ZeroSubsimplices]
        logAdjustments = [complex_to_pair(tet.cusp_triangle_vertex_positions[V][0]) for tet in self.mcomplex.Tetrahedra for V in t3m.ZeroSubsimplices]
        cuspTriangleVertexPositions = [tet.cusp_triangle_vertex_positions[V][1] for tet in self.mcomplex.Tetrahedra for V in t3m.ZeroSubsimplices]
        mat_logs = [tet.Class[V].mat_log for tet in self.mcomplex.Tetrahedra for V in t3m.ZeroSubsimplices]
        insphereRadiusParams = [tet.cosh_sqr_inradius for tet in self.mcomplex.Tetrahedra]
        isNonGeometric = self.snappy_manifold.solution_type() != 'all tetrahedra positively oriented'
        d['orientations'] = ('int[]', orientations)
        d['horosphereScales'] = ('float[]', horosphere_scales)
        d['MargulisTubes.margulisTubeTails'] = ('vec4[]', margulisTubeTails)
        d['MargulisTubes.margulisTubeHeads'] = ('vec4[]', margulisTubeHeads)
        d['margulisTubeRadiusParams'] = ('float[]', margulisTubeRadiusParams)
        d['TetCuspMatrices.cuspToTetMatrices'] = ('mat4[]', cusp_to_tet_matrices)
        d['TetCuspMatrices.tetToCuspMatrices'] = ('mat4[]', tet_to_cusp_matrices)
        d['cuspTranslations'] = ('mat2[]', cusp_translations)
        d['logAdjustments'] = ('vec2[]', logAdjustments)
        d['cuspTriangleVertexPositions'] = ('mat3x2[]', cuspTriangleVertexPositions)
        d['matLogs'] = ('mat2[]', mat_logs)
        d['insphereRadiusParams'] = ('float[]', insphereRadiusParams)
        d['isNonGeometric'] = ('bool', isNonGeometric)
        d['nonGeometricTexture'] = ('int', 0)
        return d

    def get_compile_time_constants(self):
        d = super(IdealRaytracingData, self).get_compile_time_constants()
        d[b'##finiteTrig##'] = 0
        return d

    def initial_view_state(self):
        boost = matrix([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        tet_num = 0
        weight = 0.0
        return (boost, tet_num, weight)

    def cusp_view_state_and_scale(self, which_cusp):
        vert = self.mcomplex.Vertices[which_cusp]
        corner = vert.Corners[0]
        tet = corner.Tetrahedron
        subsimplex = corner.Subsimplex
        area = self.areas[which_cusp]
        return (self.update_view_state((_cusp_view_matrix(tet, subsimplex, area), corner.Tetrahedron.Index, 0.0)), _cusp_view_scale(tet, subsimplex, area))