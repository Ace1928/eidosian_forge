from itertools import chain
import networkx as nx
from networkx.utils.decorators import not_implemented_for
Yield the articulation points, or cut vertices, of a graph.

    An articulation point or cut vertex is any node whose removal (along with
    all its incident edges) increases the number of connected components of
    a graph.  An undirected connected graph without articulation points is
    biconnected. Articulation points belong to more than one biconnected
    component of a graph.

    Notice that by convention a dyad is considered a biconnected component.

    Parameters
    ----------
    G : NetworkX Graph
        An undirected graph.

    Yields
    ------
    node
        An articulation point in the graph.

    Raises
    ------
    NetworkXNotImplemented
        If the input graph is not undirected.

    Examples
    --------

    >>> G = nx.barbell_graph(4, 2)
    >>> print(nx.is_biconnected(G))
    False
    >>> len(list(nx.articulation_points(G)))
    4
    >>> G.add_edge(2, 8)
    >>> print(nx.is_biconnected(G))
    True
    >>> len(list(nx.articulation_points(G)))
    0

    See Also
    --------
    is_biconnected
    biconnected_components
    biconnected_component_edges

    Notes
    -----
    The algorithm to find articulation points and biconnected
    components is implemented using a non-recursive depth-first-search
    (DFS) that keeps track of the highest level that back edges reach
    in the DFS tree.  A node `n` is an articulation point if, and only
    if, there exists a subtree rooted at `n` such that there is no
    back edge from any successor of `n` that links to a predecessor of
    `n` in the DFS tree.  By keeping track of all the edges traversed
    by the DFS we can obtain the biconnected components because all
    edges of a bicomponent will be traversed consecutively between
    articulation points.

    References
    ----------
    .. [1] Hopcroft, J.; Tarjan, R. (1973).
           "Efficient algorithms for graph manipulation".
           Communications of the ACM 16: 372–378. doi:10.1145/362248.362272

    