from .simplex import *
from .perm4 import Perm4, inv
class eArrow(Arrow):

    def __init__(self, tet, tail, head):
        self.Edge = comp(tail | head)
        self.Face = comp(tail)
        self.Tetrahedron = tet