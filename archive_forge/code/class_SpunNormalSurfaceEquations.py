import snappy
import FXrays
class SpunNormalSurfaceEquations:

    def __init__(self, manifold):
        self.manifold = manifold
        n = manifold.num_tetrahedra()
        self.shift_matrix = shift_matrix(n)
        gluing_equations = list(manifold.gluing_equations())
        edge_equations = Matrix(gluing_equations[:n])
        self.quad_equations = edge_equations * self.shift_matrix
        self.cusp_equations = cusp_equations = [Vector(eqn) for eqn in gluing_equations[n:]]
        slope_matrix = []
        for i in range(manifold.num_cusps()):
            slope_matrix += [-cusp_equations[2 * i + 1], cusp_equations[2 * i]]
        self.slope_matrix = Matrix(slope_matrix)

    def vertex_solutions(self, algorithm='FXrays'):
        if algorithm == 'FXrays':
            M = self.quad_equations
            return FXrays.find_Xrays(M.nrows(), M.ncols(), M.list(), 0, print_progress=False)
        elif algorithm == 'regina':
            try:
                import regina
            except ImportError:
                raise ImportError('Regina module not available')
            M = self.manifold
            T = regina.Triangulation3(M._to_string())
            ans = []
            tets = range(M.num_tetrahedra())
            if hasattr(regina.NormalSurfaces, 'enumerate'):
                surfaces = regina.NormalSurfaces.enumerate(T, regina.NS_QUAD)
            else:
                surfaces = regina.NormalSurfaces(T, regina.NS_QUAD)
            for i in range(surfaces.size()):
                S = surfaces.surface(i)
                coeff_vector = [int(S.quads(tet, quad).stringValue()) for tet in tets for quad in (1, 2, 0)]
                ans.append(coeff_vector)
            return ans
        else:
            raise ValueError("Algorithm should be one of {'FXrays', 'regina'}")

    def is_solution(self, quad_vector):
        return self.quad_equations * quad_vector == 0

    def boundary_slope_of_solution(self, quad_vector):
        return self.slope_matrix * self.shift_matrix * quad_vector