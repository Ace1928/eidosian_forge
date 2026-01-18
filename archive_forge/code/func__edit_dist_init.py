import operator
import warnings
def _edit_dist_init(len1, len2):
    lev = []
    for i in range(len1):
        lev.append([0] * len2)
    for i in range(len1):
        lev[i][0] = i
    for j in range(len2):
        lev[0][j] = j
    return lev