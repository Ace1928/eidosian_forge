from collections import Counter, defaultdict
from itertools import combinations, product
from math import inf
import networkx as nx
from networkx.utils import not_implemented_for, pairwise
def _bounded_cycle_search(G, path, length_bound):
    """The main loop of the cycle-enumeration algorithm of Gupta and Suzumura.

    Parameters
    ----------
    G : NetworkX Graph or DiGraph
       A graph

    path : list
       A cycle prefix.  All cycles generated will begin with this prefix.

    length_bound: int
        A length bound.  All cycles generated will have length at most length_bound.

    Yields
    ------
    list of nodes
       Each cycle is represented by a list of nodes along the cycle.

    References
    ----------
    .. [1] Finding All Bounded-Length Simple Cycles in a Directed Graph
       A. Gupta and T. Suzumura https://arxiv.org/abs/2105.10094

    """
    G = _NeighborhoodCache(G)
    lock = {v: 0 for v in path}
    B = defaultdict(set)
    start = path[0]
    stack = [iter(G[path[-1]])]
    blen = [length_bound]
    while stack:
        nbrs = stack[-1]
        for w in nbrs:
            if w == start:
                yield path[:]
                blen[-1] = 1
            elif len(path) < lock.get(w, length_bound):
                path.append(w)
                blen.append(length_bound)
                lock[w] = len(path)
                stack.append(iter(G[w]))
                break
        else:
            stack.pop()
            v = path.pop()
            bl = blen.pop()
            if blen:
                blen[-1] = min(blen[-1], bl)
            if bl < length_bound:
                relax_stack = [(bl, v)]
                while relax_stack:
                    bl, u = relax_stack.pop()
                    if lock.get(u, length_bound) < length_bound - bl + 1:
                        lock[u] = length_bound - bl + 1
                        relax_stack.extend(((bl + 1, w) for w in B[u].difference(path)))
            else:
                for w in G[v]:
                    B[w].add(v)