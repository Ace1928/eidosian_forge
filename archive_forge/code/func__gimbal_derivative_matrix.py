from .gimbalLoopFinder import GimbalLoopFinder
from .truncatedComplex import TruncatedComplex
from .hyperbolicStructure import HyperbolicStructure
from .verificationError import *
from sage.all import matrix, prod, RealDoubleField, pi
def _gimbal_derivative_matrix(self, gimbal_loop, path_matrices, j):

    def term(i):
        edgeLoop, path = gimbal_loop[i]
        rotation_and_derivative = self.rotations_and_derivatives_for_approx_edges[edgeLoop.edge_index]
        if i == j:
            return path_matrices[i] * rotation_and_derivative[1]
        else:
            return path_matrices[i] * rotation_and_derivative[0]
    return prod([term(i) for i in range(len(gimbal_loop) - 1, -1, -1)])