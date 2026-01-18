import networkx
import random
from .links import Strand
from ..graphs import CyclicList, Digraph
from collections import namedtuple, Counter
def _make_turn_regular(self):
    dummy = set()
    regular = [F for F in self.faces if F.is_turn_regular()]
    irregular = [F for F in self.faces if not F.is_turn_regular()]
    while len(irregular):
        F = irregular.pop()
        i, j = F.kitty_corner()
        v0, v1 = (F[i][1], F[j][1])
        kind = random.choice(('vertical', 'horizontal'))
        if len([e for e in self.incoming(v0) if e.kind == kind]):
            e = self.add_edge(v0, v1, kind)
        else:
            e = self.add_edge(v1, v0, kind)
        dummy.add(e)
        for v in [v0, v1]:
            F = OrthogonalFace(self, (e, v))
            if F.is_turn_regular():
                regular.append(F)
            else:
                irregular.append(F)
    self.faces, self.dummy = (regular, dummy)