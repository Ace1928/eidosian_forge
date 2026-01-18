from .gimbalLoopFinder import GimbalLoopFinder
from .truncatedComplex import TruncatedComplex
from .hyperbolicStructure import HyperbolicStructure
from .verificationError import *
from sage.all import matrix, prod, RealDoubleField, pi
class VerifyHyperbolicStructureEngine:

    def __init__(self, hyperbolic_structure):
        self.hyperbolic_structure = hyperbolic_structure
        self.mcomplex = self.hyperbolic_structure.mcomplex
        if not hyperbolic_structure.exact_edges:
            raise RuntimeError('Need to specify "exact edges", i.e., edges for which we already know by, e.g., the Krawczyk test that their edge equations are fullfilled')
        num_edges = len(self.mcomplex.Edges)
        exact_edges = set(hyperbolic_structure.exact_edges)
        self.approx_edges = [i for i in range(num_edges) if i not in exact_edges]
        self.truncated_complex = TruncatedComplex(self.mcomplex)
        self.gimbal_loops = [GimbalLoopFinder(self.truncated_complex, vertex, self.approx_edges).compute_grouped_loop() for vertex in self.mcomplex.Vertices]
        self.rotations_and_derivatives_for_approx_edges = {e: HyperbolicStructure.so3_matrix_and_derivative_for_z_rotation(self.hyperbolic_structure.angle_sums[e]) for e in self.approx_edges}

    def gimbal_derivative(self):
        self.edge_index_to_column_index = {e: i for i, e in enumerate(self.approx_edges)}
        RIF = self.hyperbolic_structure.vertex_gram_matrices[0].base_ring()
        num_rows = 3 * len(self.mcomplex.Vertices)
        num_cols = len(self.approx_edges)
        result = matrix(RIF, num_rows, num_cols)
        for i, gimbal_loop in enumerate(self.gimbal_loops):
            path_matrices = [self.hyperbolic_structure.so3_matrix_for_path(edgePath) for edgeLoop, edgePath in gimbal_loop]
            for j, (edgeLoop, edgePath) in enumerate(gimbal_loop):
                col = self.edge_index_to_column_index[edgeLoop.edge_index]
                m = self._gimbal_derivative_matrix(gimbal_loop, path_matrices, j)
                result[3 * i, col] += m[0, 1]
                result[3 * i + 1, col] += m[0, 2]
                result[3 * i + 2, col] += m[1, 2]
        return result

    def assert_avoiding_gimbal_lock(self):
        d = self.gimbal_derivative()
        if not VerifyHyperbolicStructureEngine.is_invertible(d):
            raise GimbalDerivativeNotInvertibleError(d, d.determinant())

    def assert_two_pi_in_ange_sum_intervals(self):
        RIF = self.hyperbolic_structure.angle_sums[0].parent()
        two_pi = RIF(2 * pi)
        for angle in self.hyperbolic_structure.angle_sums:
            if two_pi not in angle:
                raise AngleSumIntervalNotContainingTwoPiError()

    def assert_vertex_gram_matrices_valid(self):
        for G_adj in self.hyperbolic_structure.vertex_gram_adjoints:
            _assert_vertex_gram_adjoint_valid(G_adj)
        for G in self.hyperbolic_structure.vertex_gram_matrices:
            _assert_vertex_gram_matrix_valid(G)

    def assert_verified_hyperbolic(self):
        self.assert_vertex_gram_matrices_valid()
        self.assert_avoiding_gimbal_lock()
        self.assert_two_pi_in_ange_sum_intervals()

    @staticmethod
    def is_invertible(m):
        r, c = m.dimensions()
        if r != c:
            raise AssertionError('Not a square matrix in invertible check of VerifyHyperbolicStructureEngine.')
        RIF = m.base_ring()
        max_entry = RIF(1) / (r * c)
        real_m = matrix(RealDoubleField(), m)
        try:
            real_approx_inv = real_m.inverse()
        except ZeroDivisionError:
            return False
        approx_inv = real_approx_inv.change_ring(RIF)
        idMatrix = matrix.identity(RIF, r)
        t = approx_inv * m - idMatrix
        for row in t:
            for entry in row:
                if not entry < max_entry:
                    return False
        return True

    def _gimbal_derivative_matrix(self, gimbal_loop, path_matrices, j):

        def term(i):
            edgeLoop, path = gimbal_loop[i]
            rotation_and_derivative = self.rotations_and_derivatives_for_approx_edges[edgeLoop.edge_index]
            if i == j:
                return path_matrices[i] * rotation_and_derivative[1]
            else:
                return path_matrices[i] * rotation_and_derivative[0]
        return prod([term(i) for i in range(len(gimbal_loop) - 1, -1, -1)])