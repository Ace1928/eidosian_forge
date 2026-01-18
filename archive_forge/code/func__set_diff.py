import sys
import json
from .symbols import *
from .symbols import Symbol
def _set_diff(self, a, b):
    removed = a.difference(b)
    added = b.difference(a)
    if not removed and (not added):
        return ({}, 1.0)
    ranking = sorted(((self._obj_diff(x, y)[1], x, y) for x in removed for y in added), reverse=True, key=lambda x: x[0])
    r2 = set(removed)
    a2 = set(added)
    n_common = len(a) - len(removed)
    s_common = float(n_common)
    for s, x, y in ranking:
        if x in r2 and y in a2:
            r2.discard(x)
            a2.discard(y)
            s_common += s
            n_common += 1
        if not r2 or not a2:
            break
    n_tot = len(a) + len(added)
    s = s_common / n_tot if n_tot != 0 else 1.0
    return (self.options.syntax.emit_set_diff(a, b, s, added, removed), s)