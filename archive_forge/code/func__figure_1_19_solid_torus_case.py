from ..snap.t3mlite.simplex import *
from ..snap.t3mlite.edge import Edge
from ..snap.t3mlite.arrow import Arrow
from ..snap.t3mlite.mcomplex import VERBOSE
from .mcomplex_with_memory import McomplexWithMemory, edge_and_arrow
def _figure_1_19_solid_torus_case(self, a):
    """
        This is the case of Figure 1.19 in Matveev where one of the
        two tets adjacent to the valence two edge has its other two
        faces glued to itself.  See "round_1/twist.py" for details.
        """

    def standard_twist_in_back(arrow):
        a = arrow.copy()
        assert a.axis().valence() == 2
        b = a.glued()
        c = b.copy().rotate(1)
        d = c.glued()
        assert d.Tetrahedron == b.Tetrahedron
        return c.tail() == d.tail()
    if standard_twist_in_back(a):
        x_fixed = a.copy().rotate(2).glued().reverse()
        self.two_to_three(a, must_succeed=True)
        x = x_fixed.glued().opposite().rotate(2)
        self.two_to_three(x, must_succeed=True)
        x = x_fixed.glued().glued()
        self.two_to_three(x, must_succeed=True)
        e = x_fixed.glued().north_head()
        self.three_to_two(e, must_succeed=True)
        e = x_fixed.glued().glued().south_head()
        self.three_to_two(e, must_succeed=True)
        e = x_fixed.glued().north_head()
        self.three_to_two(e, must_succeed=True)
        b_new = x_fixed.glued().opposite()
        a_new = b_new.glued()
    else:
        x_fixed = a.copy().rotate(1).glued().reverse()
        self.two_to_three(a, must_succeed=True)
        x = x_fixed.glued().rotate(1).opposite()
        self.two_to_three(x, must_succeed=True)
        x = x_fixed.glued().glued()
        self.two_to_three(x, must_succeed=True)
        e = x_fixed.glued().south_head()
        self.three_to_two(e, must_succeed=True)
        e = x_fixed.glued().glued().north_head()
        self.three_to_two(e, must_succeed=True)
        e = x_fixed.glued().south_head()
        self.three_to_two(e, must_succeed=True)
        b_new = x_fixed.glued().opposite().reverse()
        a_new = b_new.glued()
    assert a_new.axis().valence() == 2
    assert b_new.axis().valence() == 2
    return (a_new, b_new)