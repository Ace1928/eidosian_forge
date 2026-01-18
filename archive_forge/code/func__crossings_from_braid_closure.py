import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
def _crossings_from_braid_closure(self, word, num_strands=None):
    """
        Compute the braid closure of a word given in the form of a list of
        integers, where 1, 2, 3, etc correspond to the generators
        sigma_1, sigma_2, sigma_3, and so on, and negative numbers to
        their inverses.

        Conventions follow the mirror image of Birman's book; if one
        views braid vertically, so that [a, b, c] corresponds to

        a
        b
        c

        then sigma_i corresponds to the rational tangle -1 in our
        conventions.

        The components of the resulting link are will be oriented
        consistently with the braid.
        """
    if num_strands is None:
        num_strands = max((abs(a) for a in word)) + 1
    strands = [Strand('s' + repr(i)) for i in range(num_strands)]
    current = [(x, 1) for x in strands]
    crossings = []
    for i, a in enumerate(word):
        C = Crossing('x' + repr(i))
        crossings.append(C)
        if a < 0:
            t0, t1 = (1, 0)
            b0, b1 = (2, 3)
        else:
            t0, t1 = (0, 3)
            b0, b1 = (1, 2)
        j0, j1 = (abs(a) - 1, abs(a))
        C.make_tail(t0)
        C.make_tail(t1)
        C.orient()
        C[t0] = current[j0]
        C[t1] = current[j1]
        current[j0] = C[b0]
        current[j1] = C[b1]
    for i in range(num_strands):
        strands[i][0] = current[i]
    return crossings + strands