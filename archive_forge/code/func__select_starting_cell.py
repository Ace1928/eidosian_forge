from collections import defaultdict
from functools import partial
from itertools import combinations
import networkx as nx
from networkx.utils import arbitrary_element
from networkx.utils.decorators import not_implemented_for
def _select_starting_cell(G, starting_edge=None):
    """Select a cell to initiate _find_partition

    Parameters
    ----------
    G : NetworkX Graph
    starting_edge: an edge to build the starting cell from

    Returns
    -------
    Tuple of vertices in G

    Raises
    ------
    NetworkXError
        If it is determined that G is not a line graph

    Notes
    -----
    If starting edge not specified then pick an arbitrary edge - doesn't
    matter which. However, this function may call itself requiring a
    specific starting edge. Note that the r, s notation for counting
    triangles is the same as in the Roussopoulos paper cited above.
    """
    if starting_edge is None:
        e = arbitrary_element(G.edges())
    else:
        e = starting_edge
        if e[0] not in G.nodes():
            raise nx.NetworkXError(f'Vertex {e[0]} not in graph')
        if e[1] not in G[e[0]]:
            msg = f'starting_edge ({e[0]}, {e[1]}) is not in the Graph'
            raise nx.NetworkXError(msg)
    e_triangles = _triangles(G, e)
    r = len(e_triangles)
    if r == 0:
        starting_cell = e
    elif r == 1:
        T = e_triangles[0]
        a, b, c = T
        ac_edges = len(_triangles(G, (a, c)))
        bc_edges = len(_triangles(G, (b, c)))
        if ac_edges == 1:
            if bc_edges == 1:
                starting_cell = T
            else:
                return _select_starting_cell(G, starting_edge=(b, c))
        else:
            return _select_starting_cell(G, starting_edge=(a, c))
    else:
        s = 0
        odd_triangles = []
        for T in e_triangles:
            if _odd_triangle(G, T):
                s += 1
                odd_triangles.append(T)
        if r == 2 and s == 0:
            starting_cell = T
        elif r - 1 <= s <= r:
            triangle_nodes = set()
            for T in odd_triangles:
                for x in T:
                    triangle_nodes.add(x)
            for u in triangle_nodes:
                for v in triangle_nodes:
                    if u != v and v not in G[u]:
                        msg = 'G is not a line graph (odd triangles do not form complete subgraph)'
                        raise nx.NetworkXError(msg)
            starting_cell = tuple(triangle_nodes)
        else:
            msg = 'G is not a line graph (incorrect number of odd triangles around starting edge)'
            raise nx.NetworkXError(msg)
    return starting_cell