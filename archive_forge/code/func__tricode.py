from collections import defaultdict
from itertools import combinations, permutations
import networkx as nx
from networkx.utils import not_implemented_for, py_random_state
def _tricode(G, v, u, w):
    """Returns the integer code of the given triad.

    This is some fancy magic that comes from Batagelj and Mrvar's paper. It
    treats each edge joining a pair of `v`, `u`, and `w` as a bit in
    the binary representation of an integer.

    """
    combos = ((v, u, 1), (u, v, 2), (v, w, 4), (w, v, 8), (u, w, 16), (w, u, 32))
    return sum((x for u, v, x in combos if v in G[u]))