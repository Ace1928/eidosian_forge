from ..snap.t3mlite.simplex import *
from ..snap.t3mlite.edge import Edge
from ..snap.t3mlite.arrow import Arrow
from ..snap.t3mlite.mcomplex import VERBOSE
from .mcomplex_with_memory import McomplexWithMemory, edge_and_arrow
def _figure_1_19_other_case(self, a):
    """
        This is the case of Figure 1.19 in Matveev where the two tets
        about the valence two edge share three faces instead of the
        usual two.
        """
    b = a.glued()
    c = a.copy().rotate(1).glued()
    assert c.Tetrahedron != a.Tetrahedron
    assert c.Tetrahedron != b.Tetrahedron

    def in_x_case(a):
        b_opp = a.glued().opposite()
        c = a.copy().opposite().glued()
        d = c.copy().rotate(1)
        e = c.copy().rotate(-1)
        assert d == b_opp or e == b_opp
        return d == b_opp
    x_fixed = a.copy().opposite().reverse().glued().reverse()
    if in_x_case(a):
        x = x_fixed.glued().opposite().reverse()
        self.two_to_three(x, must_succeed=True)
        x = x_fixed.glued().rotate(2)
        self.two_to_three(x, must_succeed=True)
        x = x_fixed.glued().glued().glued()
        self.two_to_three(x, must_succeed=True)
        e = x_fixed.glued().equator()
        self.three_to_two(e, must_succeed=True)
        e = x_fixed.glued().glued().south_head()
        self.three_to_two(e, must_succeed=True)
        e = x_fixed.glued().equator()
        self.three_to_two(e, must_succeed=True)
        self.rebuild()
        a_new = x_fixed.glued().reverse().rotate(1)
    else:
        x = x_fixed.glued().opposite()
        self.two_to_three(x, must_succeed=True)
        x = x_fixed.glued().rotate(1)
        self.two_to_three(x, must_succeed=True)
        x = x_fixed.glued().glued().glued()
        self.two_to_three(x, must_succeed=True)
        e = x_fixed.glued().equator()
        self.three_to_two(e, must_succeed=True)
        e = x_fixed.glued().glued().north_head()
        self.three_to_two(e, must_succeed=True)
        e = x_fixed.glued().equator()
        self.three_to_two(e, must_succeed=True)
        self.rebuild()
        a_new = x_fixed.glued().reverse().rotate(2).reverse()
    b_new = a_new.glued()
    assert a_new.axis() == b_new.axis() and a_new.axis().valence() == 2
    return (a_new, b_new)