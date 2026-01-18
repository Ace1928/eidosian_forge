from .links import CrossingStrand
from ..graphs import CyclicList
class TwistRegionCrossing(TwistRegionCap):
    """
    A crossing together with a chosen bigon.  Recorded by the CrossingStrand
    which gives the side of said bigon which is "most clockwise" when viewed from
    the crossing.
    """

    def __init__(self, crossing):
        if isinstance(crossing, CrossingStrand):
            self.cs = crossing
        else:
            assert is_end_of_twist_region(crossing)
            neighbors = CyclicList((C for C, i in crossing.adjacent))
            for i in range(4):
                if neighbors[i] == neighbors[i + 1]:
                    self.cs = CrossingStrand(crossing, i)

    def next(self):
        cs = self.cs.opposite().rotate(1)
        C, e = cs
        if C.adjacent[e][0] != C.adjacent[e + 1][0]:
            return TwistRegionCap(cs)
        return TwistRegionCrossing(cs)

    def __repr__(self):
        return '<Twist: %s>' % (self.cs,)