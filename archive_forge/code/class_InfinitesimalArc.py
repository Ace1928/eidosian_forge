from ..snap.t3mlite.simplex import *
from .rational_linear_algebra import Matrix, Vector3, Vector4
from . import pl_utils
class InfinitesimalArc(Arc):
    """
    A length 0 arc corresponding to moving across a face from one
    tetrahedron to the adjacent one.
    """

    def __init__(self, start, end, start_tet, end_tet, past=None, next=None):
        self.start, self.end = (start, end)
        self.start_tet, self.end_tet = (start_tet, end_tet)
        self.past, self.next = (past, next)

    def __repr__(self):
        v, i = (self.start, self.start_tet)
        w, j = (self.end, self.end_tet)
        return f'InfArc({i}:{v}; {j}:{w})'