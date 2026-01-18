from collections import defaultdict
from functools import partial
from itertools import combinations
import networkx as nx
from networkx.utils import arbitrary_element
from networkx.utils.decorators import not_implemented_for
def _lg_directed(G, create_using=None):
    """Returns the line graph L of the (multi)digraph G.

    Edges in G appear as nodes in L, represented as tuples of the form (u,v)
    or (u,v,key) if G is a multidigraph. A node in L corresponding to the edge
    (u,v) is connected to every node corresponding to an edge (v,w).

    Parameters
    ----------
    G : digraph
        A directed graph or directed multigraph.
    create_using : NetworkX graph constructor, optional
       Graph type to create. If graph instance, then cleared before populated.
       Default is to use the same graph class as `G`.

    """
    L = nx.empty_graph(0, create_using, default=G.__class__)
    get_edges = partial(G.edges, keys=True) if G.is_multigraph() else G.edges
    for from_node in get_edges():
        L.add_node(from_node)
        for to_node in get_edges(from_node[1]):
            L.add_edge(from_node, to_node)
    return L