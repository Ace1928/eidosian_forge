import functools
import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils.decorators import not_implemented_for
Incremental closeness centrality for nodes.

    Compute closeness centrality for nodes using level-based work filtering
    as described in Incremental Algorithms for Closeness Centrality by Sariyuce et al.

    Level-based work filtering detects unnecessary updates to the closeness
    centrality and filters them out.

    ---
    From "Incremental Algorithms for Closeness Centrality":

    Theorem 1: Let :math:`G = (V, E)` be a graph and u and v be two vertices in V
    such that there is no edge (u, v) in E. Let :math:`G' = (V, E \cup uv)`
    Then :math:`cc[s] = cc'[s]` if and only if :math:`\left|dG(s, u) - dG(s, v)\right| \leq 1`.

    Where :math:`dG(u, v)` denotes the length of the shortest path between
    two vertices u, v in a graph G, cc[s] is the closeness centrality for a
    vertex s in V, and cc'[s] is the closeness centrality for a
    vertex s in V, with the (u, v) edge added.
    ---

    We use Theorem 1 to filter out updates when adding or removing an edge.
    When adding an edge (u, v), we compute the shortest path lengths from all
    other nodes to u and to v before the node is added. When removing an edge,
    we compute the shortest path lengths after the edge is removed. Then we
    apply Theorem 1 to use previously computed closeness centrality for nodes
    where :math:`\left|dG(s, u) - dG(s, v)\right| \leq 1`. This works only for
    undirected, unweighted graphs; the distance argument is not supported.

    Closeness centrality [1]_ of a node `u` is the reciprocal of the
    sum of the shortest path distances from `u` to all `n-1` other nodes.
    Since the sum of distances depends on the number of nodes in the
    graph, closeness is normalized by the sum of minimum possible
    distances `n-1`.

    .. math::

        C(u) = \frac{n - 1}{\sum_{v=1}^{n-1} d(v, u)},

    where `d(v, u)` is the shortest-path distance between `v` and `u`,
    and `n` is the number of nodes in the graph.

    Notice that higher values of closeness indicate higher centrality.

    Parameters
    ----------
    G : graph
      A NetworkX graph

    edge : tuple
      The modified edge (u, v) in the graph.

    prev_cc : dictionary
      The previous closeness centrality for all nodes in the graph.

    insertion : bool, optional
      If True (default) the edge was inserted, otherwise it was deleted from the graph.

    wf_improved : bool, optional (default=True)
      If True, scale by the fraction of nodes reachable. This gives the
      Wasserman and Faust improved formula. For single component graphs
      it is the same as the original formula.

    Returns
    -------
    nodes : dictionary
      Dictionary of nodes with closeness centrality as the value.

    See Also
    --------
    betweenness_centrality, load_centrality, eigenvector_centrality,
    degree_centrality, closeness_centrality

    Notes
    -----
    The closeness centrality is normalized to `(n-1)/(|G|-1)` where
    `n` is the number of nodes in the connected part of graph
    containing the node.  If the graph is not completely connected,
    this algorithm computes the closeness centrality for each
    connected part separately.

    References
    ----------
    .. [1] Freeman, L.C., 1979. Centrality in networks: I.
       Conceptual clarification.  Social Networks 1, 215--239.
       https://doi.org/10.1016/0378-8733(78)90021-7
    .. [2] Sariyuce, A.E. ; Kaya, K. ; Saule, E. ; Catalyiirek, U.V. Incremental
       Algorithms for Closeness Centrality. 2013 IEEE International Conference on Big Data
       http://sariyuce.com/papers/bigdata13.pdf
    