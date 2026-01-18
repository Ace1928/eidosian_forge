import random
from .links_base import CrossingEntryPoint
from ..sage_helper import _within_sage
class Exhaustion:
    """
    An exhaustion of a link where crossings are added in one-by-one
    so that the resulting tangle is connected at every stage.

    Starting at the given crossing, it uses a greedy algorithm to try
    to minimize the sizes of the frontiers of the intermediate tangles.

    If no initial crossing is specified, one is chosen at random.
    """

    def __init__(self, link, crossing=None):
        if crossing is None:
            crossing = random.choice(link.crossings)
        crossings = [crossing]
        gluings = [[]]
        frontier = set(crossing.crossing_strands())
        frontier_lengths = [4]
        while len(crossings) < len(link.crossings):
            choices = [(num_overlap(cs.opposite()[0], frontier), cs) for cs in frontier]
            overlap, cs = max(choices, key=lambda x: x[0])
            C = cs.opposite().crossing
            assert C not in crossings
            crossings.append(C)
            C_gluings = []
            for cs in C.crossing_strands():
                opp = cs.opposite()
                if opp in frontier:
                    frontier.discard(opp)
                    b = cs.oriented()
                    a = b.opposite()
                    C_gluings.append((a, b))
                else:
                    frontier.add(cs)
            assert frontier_lengths[-1] + 4 - 2 * overlap == len(frontier)
            frontier_lengths.append(len(frontier))
            gluings.append(C_gluings)
        self.link = link
        self.crossings = crossings
        self.frontier_lengths = frontier_lengths
        self.gluings = gluings
        self.width = max(frontier_lengths) // 2

    def test_indices(self):
        indices = StrandIndices(self.link, self.crossings)
        all_gluings = sum(self.gluings, [])[:-1]
        for a, b in all_gluings[:-1]:
            indices.merge(a, b)

    def alexander_polynomial(self):
        D = DrorDatum(self.link, self.crossings)
        gluings = self.gluings[:]
        for C, gluings in list(zip(self.crossings, gluings))[:-1]:
            D.add_crossing(C)
            for a, b in gluings:
                D.merge(a, b)
        C = self.crossings[-1]
        D.add_crossing(C)
        for a, b in self.gluings[-1][:-1]:
            D.merge(a, b)
        alex = D.omega
        p, q = (alex.numerator(), alex.denominator())
        assert [abs(c) for c in q.coefficients()] == [1]
        if p.leading_coefficient() < 0:
            p = -p
        t, e = (p.parent().gen(), min(p.exponents()))
        return p // t ** e