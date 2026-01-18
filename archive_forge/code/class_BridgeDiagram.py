from ..sage_helper import _within_sage
from ..graphs import CyclicList, Digraph
from .links import CrossingStrand, Crossing, Strand, Link
from .orthogonal import basic_topological_numbering
from .tangles import join_strands, RationalTangle
class BridgeDiagram:
    """
    A proper bridge diagram of a link, that is, a height function
    where all the mins are below all the maxes.
    """

    def __init__(self, bottom, crossings, top):
        self.bottom, self.crossings, self.top = (bottom, crossings, top)
        self.width = 2 * len(bottom)
        self.name = 'None'

    def link(self):
        crossings = []
        curr_endpoints = self.width * [None]
        for x, y in self.bottom:
            s = Strand()
            crossings.append(s)
            curr_endpoints[x] = (s, 0)
            curr_endpoints[y] = (s, 1)
        for a, b in self.crossings:
            c = Crossing()
            crossings.append(c)
            if a < b:
                ins, outs = ((3, 0), (2, 1))
            else:
                ins, outs = ((0, 1), (3, 2))
                a, b = (b, a)
            c[ins[0]] = curr_endpoints[a]
            c[ins[1]] = curr_endpoints[b]
            curr_endpoints[a] = (c, outs[0])
            curr_endpoints[b] = (c, outs[1])
        for x, y in self.top:
            join_strands(curr_endpoints[x], curr_endpoints[y])
        return Link(crossings)

    def bohua_code(self):
        b = self.width // 2
        ans = [b]
        ans += pairing_to_permuation(self.bottom)
        ans += [len(self.crossings)] + list(sum(self.crossings, tuple()))
        ans += pairing_to_permuation(self.top)
        return self.name + '\t' + ' '.join((repr(a) for a in ans))

    def HF(self):
        import bohua_HF
        return bohua_HF.compute_HF(self.bohua_code())

    def is_proper(self):
        return all((abs(a - b) < 2 for a, b in self.crossings))