from .simplex import *
from .tetrahedron import Tetrahedron
from .corner import Corner
from .arrow import Arrow
from .face import Face
from .edge import Edge
from .vertex import Vertex
from .surface import Surface, SpunSurface, ClosedSurface, ClosedSurfaceInCusped
from .perm4 import Perm4, inv
from . import files
from . import linalg
from . import homology
import sys
import random
import io
def add_fan(self, edge, n):
    """
        Adds a fan of ``n`` tetrahedra onto a boundary edge and rebuilds.
        """
    if not edge.IntOrBdry == 'bdry':
        return 0
    a = edge.LeftBdryArrow
    b = edge.RightBdryArrow.reverse()
    if n == 0:
        a.glue(b)
        return 1
    new = self.new_arrows(n)
    a.glue(new[0])
    for j in range(len(new) - 1):
        new[j].glue(new[j + 1])
    new[-1].glue(b)
    self.rebuild()
    return 1