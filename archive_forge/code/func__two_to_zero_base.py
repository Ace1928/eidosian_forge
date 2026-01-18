from ..snap.t3mlite.simplex import *
from ..snap.t3mlite.edge import Edge
from ..snap.t3mlite.arrow import Arrow
from ..snap.t3mlite.mcomplex import VERBOSE
from .mcomplex_with_memory import McomplexWithMemory, edge_and_arrow
def _two_to_zero_base(self, arrow):
    """
        The easiest case of the 2->0 move is when there is a
        tetrahedron with a pair of opposite edges each of which has
        valence two.  In Matveev, this is the V^-1 move.  It is a
        implemented as a compound of four 2<->3 moves.
        """
    assert arrow.axis().valence() == 2
    assert arrow.equator().valence() == 2
    a = arrow.copy()
    a.opposite().next().reverse().opposite()
    if a.Tetrahedron == a.glued().Tetrahedron:
        arrow = arrow.copy().opposite()
        a = arrow.copy()
        a.opposite().next().reverse().opposite()
        if a.Tetrahedron == a.glued().Tetrahedron:
            assert len(self.Tetrahedra) == 3
            raise ValueError('Cannot do this 0->2 move via 2<->3 moves')
    elif a.glued().Tetrahedron == arrow.glued().Tetrahedron:
        a.reverse()
        if a.glued().Tetrahedron == arrow.glued().Tetrahedron:
            assert len(self.Tetrahedra) == 3
            raise ValueError('Cannot do this 0->2 move via 2<->3 moves')
    e = self.two_to_three(a, return_arrow=True, must_succeed=True)
    b = self.three_to_two(e, return_arrow=True, must_succeed=True)
    e = b.copy().next().reverse()
    b = self.three_to_two(e, return_arrow=True, must_succeed=True)
    e = b.opposite().reverse().next().north_tail()
    self.three_to_two(e)
    return True