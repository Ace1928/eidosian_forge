from collections import Counter, defaultdict
from itertools import combinations, product
from math import inf
import networkx as nx
from networkx.utils import not_implemented_for, pairwise
def _undirected_cycle_search(G, length_bound):
    """A dispatch function for `simple_cycles` for undirected graphs.

    We generate all cycles of G through binary partition.

        1. Pick an edge (u, v) in G which belongs to at least one cycle
            a. Generate all cycles of G which contain the edge (u, v)
            b. Recursively generate all cycles of G \\ (u, v)

    This is accomplished through the following:

        1. Compute the biconnected components BCC of G.
        2. Select and remove a biconnected component C from BCC.  Select a
           non-tree edge (u, v) of a depth-first search of G[C].
        3. For each (v -> u) path P remaining in G[C] \\ (u, v), yield P.
        4. Add the biconnected components of G[C] \\ (u, v) to BCC.

    If the parameter length_bound is not None, then step 3 will be limited to simple paths
    of length at most length_bound.

    Parameters
    ----------
    G : NetworkX Graph
       An undirected graph

    length_bound : int or None
       If length_bound is an int, generate all simple cycles of G with length at most length_bound.
       Otherwise, generate all simple cycles of G.

    Yields
    ------
    list of nodes
       Each cycle is represented by a list of nodes along the cycle.
    """
    bcc = nx.biconnected_components
    components = [c for c in bcc(G) if len(c) >= 3]
    while components:
        c = components.pop()
        Gc = G.subgraph(c)
        uv = list(next(iter(Gc.edges)))
        G.remove_edge(*uv)
        if length_bound is None:
            yield from _johnson_cycle_search(Gc, uv)
        else:
            yield from _bounded_cycle_search(Gc, uv, length_bound)
        components.extend((c for c in bcc(Gc) if len(c) >= 3))