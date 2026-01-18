from .links import CrossingStrand
from ..graphs import CyclicList
class TwistRegionCap:

    def __init__(self, crossing_strand):
        self.cs = crossing_strand

    def sign(self):
        return self.cs.strand_index % 2

    def swap_crossing(self):
        self.cs.crossing.rotate_by_90()