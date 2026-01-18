from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
def _add_edge_dict(self):
    """
        Adds a dictionary that maps a pair of vertices to all edges
        of the triangulation connecting these vertices.
        The key is a pair (v0, v1) of integers with v0 < v1 that are the
        indices of the two vertices.
        """
    self._edge_dict = {}
    for edge in self.mcomplex.Edges:
        vert0, vert1 = edge.Vertices
        key = tuple(sorted([vert0.Index, vert1.Index]))
        self._edge_dict.setdefault(key, []).append(edge)