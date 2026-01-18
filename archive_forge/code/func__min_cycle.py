from collections import Counter, defaultdict
from itertools import combinations, product
from math import inf
import networkx as nx
from networkx.utils import not_implemented_for, pairwise
def _min_cycle(G, orth, weight):
    """
    Computes the minimum weight cycle in G,
    orthogonal to the vector orth as per [p. 338, 1]
    Use (u, 1) to indicate the lifted copy of u (denoted u' in paper).
    """
    Gi = nx.Graph()
    for u, v, wt in G.edges(data=weight, default=1):
        if (u, v) in orth or (v, u) in orth:
            Gi.add_edges_from([(u, (v, 1)), ((u, 1), v)], Gi_weight=wt)
        else:
            Gi.add_edges_from([(u, v), ((u, 1), (v, 1))], Gi_weight=wt)
    spl = nx.shortest_path_length
    lift = {n: spl(Gi, source=n, target=(n, 1), weight='Gi_weight') for n in G}
    start = min(lift, key=lift.get)
    end = (start, 1)
    min_path_i = nx.shortest_path(Gi, source=start, target=end, weight='Gi_weight')
    min_path = [n if n in G else n[0] for n in min_path_i]
    edgelist = list(pairwise(min_path))
    edgeset = set()
    for e in edgelist:
        if e in edgeset:
            edgeset.remove(e)
        elif e[::-1] in edgeset:
            edgeset.remove(e[::-1])
        else:
            edgeset.add(e)
    min_edgelist = []
    for e in edgelist:
        if e in edgeset:
            min_edgelist.append(e)
            edgeset.remove(e)
        elif e[::-1] in edgeset:
            min_edgelist.append(e[::-1])
            edgeset.remove(e[::-1])
    return min_edgelist