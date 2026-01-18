from .simplex import *
from .corner import Corner
from .arrow import Arrow
from .perm4 import Perm4
import sys
def _add_corner(self, arrow):
    """
        Used by Mcomplex.build_edge_classes
        """
    self.Corners.append(Corner(arrow.Tetrahedron, arrow.Edge))
    tail, head = _edge_add_corner_dict[arrow.Edge, arrow.Face]
    self._edge_orient_cache[arrow.Tetrahedron, tail, head] = 1
    self._edge_orient_cache[arrow.Tetrahedron, head, tail] = -1