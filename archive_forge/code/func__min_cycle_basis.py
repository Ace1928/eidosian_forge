from collections import Counter, defaultdict
from itertools import combinations, product
from math import inf
import networkx as nx
from networkx.utils import not_implemented_for, pairwise
def _min_cycle_basis(G, weight):
    cb = []
    tree_edges = list(nx.minimum_spanning_edges(G, weight=None, data=False))
    chords = G.edges - tree_edges - {(v, u) for u, v in tree_edges}
    set_orth = [{edge} for edge in chords]
    while set_orth:
        base = set_orth.pop()
        cycle_edges = _min_cycle(G, base, weight)
        cb.append([v for u, v in cycle_edges])
        set_orth = [{e for e in orth if e not in base if e[::-1] not in base} | {e for e in base if e not in orth if e[::-1] not in orth} if sum((e in orth or e[::-1] in orth for e in cycle_edges)) % 2 else orth for orth in set_orth]
    return cb