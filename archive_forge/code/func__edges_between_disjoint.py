import itertools as it
import math
from collections import defaultdict, namedtuple
import networkx as nx
from networkx.utils import not_implemented_for, py_random_state
def _edges_between_disjoint(H, only1, only2):
    """finds edges between disjoint nodes"""
    only1_adj = {u: set(H.adj[u]) for u in only1}
    for u, neighbs in only1_adj.items():
        neighbs12 = neighbs.intersection(only2)
        for v in neighbs12:
            yield (u, v)