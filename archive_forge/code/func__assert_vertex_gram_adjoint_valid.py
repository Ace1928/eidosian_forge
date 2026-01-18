from .gimbalLoopFinder import GimbalLoopFinder
from .truncatedComplex import TruncatedComplex
from .hyperbolicStructure import HyperbolicStructure
from .verificationError import *
from sage.all import matrix, prod, RealDoubleField, pi
def _assert_vertex_gram_adjoint_valid(G_adj):
    for i in range(4):
        if not G_adj[i, i] < 0:
            raise BadVertexGramMatrixError('Failed to verify diagonal %d cofactor < 0' % i)
        for j in range(4):
            if i != j:
                if not G_adj[i, j] ** 2 < G_adj[i, i] * G_adj[j, j]:
                    raise BadVertexGramMatrixError('Failed to verify cofactor inequality for (%d,%d)' % (i, j))