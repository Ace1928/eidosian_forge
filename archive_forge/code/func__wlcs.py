import nltk
import os
import re
import itertools
import collections
import pkg_resources
def _wlcs(x, y, weight_factor):
    m = len(x)
    n = len(y)
    vals = collections.defaultdict(float)
    dirs = collections.defaultdict(int)
    lengths = collections.defaultdict(int)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                length_tmp = lengths[i - 1, j - 1]
                vals[i, j] = vals[i - 1, j - 1] + (length_tmp + 1) ** weight_factor - length_tmp ** weight_factor
                dirs[i, j] = '|'
                lengths[i, j] = length_tmp + 1
            elif vals[i - 1, j] >= vals[i, j - 1]:
                vals[i, j] = vals[i - 1, j]
                dirs[i, j] = '^'
                lengths[i, j] = 0
            else:
                vals[i, j] = vals[i, j - 1]
                dirs[i, j] = '<'
                lengths[i, j] = 0
    return (vals, dirs)