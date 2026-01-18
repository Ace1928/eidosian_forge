from ..snap.t3mlite.simplex import *
from ..snap.t3mlite.edge import Edge
from ..snap.t3mlite.arrow import Arrow
from ..snap.t3mlite.mcomplex import Mcomplex, VERBOSE, edge_and_arrow
from ..snap.t3mlite.tetrahedron import Tetrahedron
def _add_to_move_memory(self, move, arrow):
    if hasattr(self, 'move_memory'):
        T = arrow.Tetrahedron
        tet_index = self.Tetrahedra.index(T)
        self.move_memory.append((move, arrow.Edge, arrow.Face, tet_index))