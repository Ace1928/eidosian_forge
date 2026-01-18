from .links import CrossingStrand
from ..graphs import CyclicList
class TwistRegion:

    def __init__(self, crossing):
        C = TwistRegionCrossing(crossing)
        crossings = [C]
        while isinstance(C, TwistRegionCrossing):
            C = C.next()
            crossings.append(C)
        self.crossings = crossings

    def signs(self):
        return [C.sign() for C in self.crossings]

    def make_consistent(self):
        sign = self.crossings[0].sign()
        for C in self.crossings:
            if C.sign() != sign:
                C.swap_crossing()

    def __len__(self):
        return len(self.crossings)