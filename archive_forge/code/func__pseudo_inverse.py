from .hyperbolicStructure import *
from .verificationError import *
from sage.all import RDF, pi, matrix, block_matrix, vector
def _pseudo_inverse(m, verbose=False):
    global _singular_epsilon
    u, d, v = m.SVD()
    dims = d.dimensions()
    dQuasiInverse = matrix(RDF, dims[1], dims[0])
    rank = 0
    for i in range(min(dims)):
        if abs(d[i, i]) > _singular_epsilon:
            dQuasiInverse[i, i] = 1.0 / d[i, i]
            rank += 1
    if verbose:
        print('Rank: %d' % rank)
    return v * dQuasiInverse * u.transpose()