import operator
import warnings
def _edit_dist_step(lev, i, j, s1, s2, last_left, last_right, substitution_cost=1, transpositions=False):
    c1 = s1[i - 1]
    c2 = s2[j - 1]
    a = lev[i - 1][j] + 1
    b = lev[i][j - 1] + 1
    c = lev[i - 1][j - 1] + (substitution_cost if c1 != c2 else 0)
    d = c + 1
    if transpositions and last_left > 0 and (last_right > 0):
        d = lev[last_left - 1][last_right - 1] + i - last_left + j - last_right - 1
    lev[i][j] = min(a, b, c, d)