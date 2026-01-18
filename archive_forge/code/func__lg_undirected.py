from collections import defaultdict
from functools import partial
from itertools import combinations
import networkx as nx
from networkx.utils import arbitrary_element
from networkx.utils.decorators import not_implemented_for
def _lg_undirected(G, selfloops=False, create_using=None):
    """Returns the line graph L of the (multi)graph G.

    Edges in G appear as nodes in L, represented as sorted tuples of the form
    (u,v), or (u,v,key) if G is a multigraph. A node in L corresponding to
    the edge {u,v} is connected to every node corresponding to an edge that
    involves u or v.

    Parameters
    ----------
    G : graph
        An undirected graph or multigraph.
    selfloops : bool
        If `True`, then self-loops are included in the line graph. If `False`,
        they are excluded.
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Notes
    -----
    The standard algorithm for line graphs of undirected graphs does not
    produce self-loops.

    """
    L = nx.empty_graph(0, create_using, default=G.__class__)
    get_edges = partial(G.edges, keys=True) if G.is_multigraph() else G.edges
    shift = 0 if selfloops else 1
    node_index = {n: i for i, n in enumerate(G)}
    edge_key_function = lambda edge: (node_index[edge[0]], node_index[edge[1]])
    edges = set()
    for u in G:
        nodes = [tuple(sorted(x[:2], key=node_index.get)) + x[2:] for x in get_edges(u)]
        if len(nodes) == 1:
            L.add_node(nodes[0])
        for i, a in enumerate(nodes):
            edges.update([tuple(sorted((a, b), key=edge_key_function)) for b in nodes[i + shift:]])
    L.add_edges_from(edges)
    return L