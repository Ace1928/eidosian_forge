from .gimbalLoopFinder import GimbalLoopFinder
from .truncatedComplex import TruncatedComplex
from .hyperbolicStructure import HyperbolicStructure
from .verificationError import *
from sage.all import matrix, prod, RealDoubleField, pi
def _assert_vertex_gram_matrix_valid(G):
    for i in range(4):
        for j in range(4):
            if not G[i, j] <= -1:
                raise BadVertexGramMatrixError('Failed to verify entry (%d,%d) <= - 1' % (i, j))
    p = G.characteristic_polynomial()
    a0, a1, a2, a3, a4 = p.coefficients(sparse=False)
    if not a3 == 4:
        raise BadVertexGramMatrixError('Failed to verify a3 = 4 for characteristic polynomial')
    if not a4 == 1:
        raise BadVertexGramMatrixError('Failed to verify a4 = 1 for characteristic polynomial')
    if not a0 < 0:
        raise BadVertexGramMatrixError('Failed to verify a0 < 0 for characteristic polynomial')
    if not a1 > 0:
        raise BadVertexGramMatrixError('Failed to verify a1 > 0 for characteristic polynomial')
    if not a2 < 0:
        raise BadVertexGramMatrixError('Failed to verify a2 < 0 for characteristic polynomial')