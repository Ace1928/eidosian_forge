from collections import Counter, defaultdict
from itertools import combinations, product
from math import inf
import networkx as nx
from networkx.utils import not_implemented_for, pairwise
def _johnson_cycle_search(G, path):
    """The main loop of the cycle-enumeration algorithm of Johnson.

    Parameters
    ----------
    G : NetworkX Graph or DiGraph
       A graph

    path : list
       A cycle prefix.  All cycles generated will begin with this prefix.

    Yields
    ------
    list of nodes
       Each cycle is represented by a list of nodes along the cycle.

    References
    ----------
        .. [1] Finding all the elementary circuits of a directed graph.
       D. B. Johnson, SIAM Journal on Computing 4, no. 1, 77-84, 1975.
       https://doi.org/10.1137/0204007

    """
    G = _NeighborhoodCache(G)
    blocked = set(path)
    B = defaultdict(set)
    start = path[0]
    stack = [iter(G[path[-1]])]
    closed = [False]
    while stack:
        nbrs = stack[-1]
        for w in nbrs:
            if w == start:
                yield path[:]
                closed[-1] = True
            elif w not in blocked:
                path.append(w)
                closed.append(False)
                stack.append(iter(G[w]))
                blocked.add(w)
                break
        else:
            stack.pop()
            v = path.pop()
            if closed.pop():
                if closed:
                    closed[-1] = True
                unblock_stack = {v}
                while unblock_stack:
                    u = unblock_stack.pop()
                    if u in blocked:
                        blocked.remove(u)
                        unblock_stack.update(B[u])
                        B[u].clear()
            else:
                for w in G[v]:
                    B[w].add(v)