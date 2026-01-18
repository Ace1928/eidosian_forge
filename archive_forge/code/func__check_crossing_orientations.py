import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
def _check_crossing_orientations(self):
    for C in self.crossings:
        if C.sign == 1:
            assert C.directions == {(0, 2), (3, 1)}
        elif C.sign == -1:
            assert C.directions == {(0, 2), (1, 3)}
        else:
            assert False
        for a, b in C.directions:
            D, d = C.adjacent[b]
            assert d in {x for x, y in D.directions}
            D, d = C.adjacent[a]
            assert d in {y for x, y in D.directions}