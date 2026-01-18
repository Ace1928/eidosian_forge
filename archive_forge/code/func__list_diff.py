import sys
import json
from .symbols import *
from .symbols import Symbol
def _list_diff(self, X, Y):
    m = len(X)
    n = len(Y)
    C = [[0 for j in range(n + 1)] for i in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            _, s = self._obj_diff(X[i - 1], Y[j - 1])
            C[i][j] = max(C[i][j - 1], C[i - 1][j], C[i - 1][j - 1] + s)
    inserted = []
    deleted = []
    changed = {}
    tot_s = 0.0
    for sign, value, pos, s in self._list_diff_0(C, X, Y):
        if sign == 1:
            inserted.append((pos, value))
        elif sign == -1:
            deleted.insert(0, (pos, value))
        elif sign == 0 and s < 1:
            changed[pos] = value
        tot_s += s
    tot_n = len(X) + len(inserted)
    if tot_n == 0:
        s = 1.0
    else:
        s = tot_s / tot_n
    return (self.options.syntax.emit_list_diff(X, Y, s, inserted, changed, deleted), s)