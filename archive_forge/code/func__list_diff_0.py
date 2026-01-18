import sys
import json
from .symbols import *
from .symbols import Symbol
def _list_diff_0(self, C, X, Y):
    i, j = (len(X), len(Y))
    r = []
    while True:
        if i > 0 and j > 0:
            d, s = self._obj_diff(X[i - 1], Y[j - 1])
            if s > 0 and C[i][j] == C[i - 1][j - 1] + s:
                r.append((0, d, j - 1, s))
                i, j = (i - 1, j - 1)
                continue
        if j > 0 and (i == 0 or C[i][j - 1] >= C[i - 1][j]):
            r.append((1, Y[j - 1], j - 1, 0.0))
            j = j - 1
            continue
        if i > 0 and (j == 0 or C[i][j - 1] < C[i - 1][j]):
            r.append((-1, X[i - 1], i - 1, 0.0))
            i = i - 1
            continue
        return reversed(r)