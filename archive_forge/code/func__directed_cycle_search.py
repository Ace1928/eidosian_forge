from collections import Counter, defaultdict
from itertools import combinations, product
from math import inf
import networkx as nx
from networkx.utils import not_implemented_for, pairwise
def _directed_cycle_search(G, length_bound):
    """A dispatch function for `simple_cycles` for directed graphs.

    We generate all cycles of G through binary partition.

        1. Pick a node v in G which belongs to at least one cycle
            a. Generate all cycles of G which contain the node v.
            b. Recursively generate all cycles of G \\ v.

    This is accomplished through the following:

        1. Compute the strongly connected components SCC of G.
        2. Select and remove a biconnected component C from BCC.  Select a
           non-tree edge (u, v) of a depth-first search of G[C].
        3. For each simple cycle P containing v in G[C], yield P.
        4. Add the biconnected components of G[C \\ v] to BCC.

    If the parameter length_bound is not None, then step 3 will be limited to
    simple cycles of length at most length_bound.

    Parameters
    ----------
    G : NetworkX DiGraph
       A directed graph

    length_bound : int or None
       If length_bound is an int, generate all simple cycles of G with length at most length_bound.
       Otherwise, generate all simple cycles of G.

    Yields
    ------
    list of nodes
       Each cycle is represented by a list of nodes along the cycle.
    """
    scc = nx.strongly_connected_components
    components = [c for c in scc(G) if len(c) >= 2]
    while components:
        c = components.pop()
        Gc = G.subgraph(c)
        v = next(iter(c))
        if length_bound is None:
            yield from _johnson_cycle_search(Gc, [v])
        else:
            yield from _bounded_cycle_search(Gc, [v], length_bound)
        G.remove_node(v)
        components.extend((c for c in scc(Gc) if len(c) >= 2))