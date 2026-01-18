import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
def _orient_crossings(self, start_orientations=None):
    if self.all_crossings_oriented():
        return
    if start_orientations is None:
        start_orientations = list()
    remaining = OrderedSet([(c, i) for c in self.crossings for i in range(4) if c.sign == 0])
    while len(remaining):
        if len(start_orientations) > 0:
            c, i = start = start_orientations.pop()
        else:
            c, i = start = remaining.pop()
        finished = False
        while not finished:
            d, j = c.adjacent[i]
            d.make_tail(j)
            (remaining.discard((c, i)), remaining.discard((d, j)))
            c, i = (d, (j + 2) % 4)
            finished = (c, i) == start
    for c in self.crossings:
        c.orient()