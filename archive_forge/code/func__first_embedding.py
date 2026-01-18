from .simplex import *
from .corner import Corner
from .arrow import Arrow
from .perm4 import Perm4
import sys
def _first_embedding(self):
    """
        For this edge, return an edge embedding similar
        to regina, that is a pair (tetrahedron, permutation) such that
        vertex 0 and 1 of the tetrahedron span the edge.
        """
    corner = self.Corners[0]
    tet = corner.Tetrahedron
    for perm in Perm4.A4():
        if corner.Subsimplex == perm.image(E01):
            if tet.Class[perm.image(V0)] == self.Vertices[0]:
                if tet.Class[perm.image(V1)] == self.Vertices[1]:
                    return (tet, perm)