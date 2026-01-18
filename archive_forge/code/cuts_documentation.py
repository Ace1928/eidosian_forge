import itertools
import networkx as nx
from networkx.algorithms.flow import build_residual_network, edmonds_karp
from .utils import build_auxiliary_edge_connectivity, build_auxiliary_node_connectivity
Returns a set of edges of minimum cardinality that disconnects G.

    If source and target nodes are provided, this function returns the
    set of edges of minimum cardinality that, if removed, would break
    all paths among source and target in G. If not, it returns a set of
    edges of minimum cardinality that disconnects G.

    Parameters
    ----------
    G : NetworkX graph

    s : node
        Source node. Optional. Default value: None.

    t : node
        Target node. Optional. Default value: None.

    flow_func : function
        A function for computing the maximum flow among a pair of nodes.
        The function has to accept at least three parameters: a Digraph,
        a source node, and a target node. And return a residual network
        that follows NetworkX conventions (see :meth:`maximum_flow` for
        details). If flow_func is None, the default maximum flow function
        (:meth:`edmonds_karp`) is used. See below for details. The
        choice of the default function may change from version
        to version and should not be relied on. Default value: None.

    Returns
    -------
    cutset : set
        Set of edges that, if removed, would disconnect G. If source
        and target nodes are provided, the set contains the edges that
        if removed, would destroy all paths between source and target.

    Examples
    --------
    >>> # Platonic icosahedral graph has edge connectivity 5
    >>> G = nx.icosahedral_graph()
    >>> len(nx.minimum_edge_cut(G))
    5

    You can use alternative flow algorithms for the underlying
    maximum flow computation. In dense networks the algorithm
    :meth:`shortest_augmenting_path` will usually perform better
    than the default :meth:`edmonds_karp`, which is faster for
    sparse networks with highly skewed degree distributions.
    Alternative flow functions have to be explicitly imported
    from the flow package.

    >>> from networkx.algorithms.flow import shortest_augmenting_path
    >>> len(nx.minimum_edge_cut(G, flow_func=shortest_augmenting_path))
    5

    If you specify a pair of nodes (source and target) as parameters,
    this function returns the value of local edge connectivity.

    >>> nx.edge_connectivity(G, 3, 7)
    5

    If you need to perform several local computations among different
    pairs of nodes on the same graph, it is recommended that you reuse
    the data structures used in the maximum flow computations. See
    :meth:`local_edge_connectivity` for details.

    Notes
    -----
    This is a flow based implementation of minimum edge cut. For
    undirected graphs the algorithm works by finding a 'small' dominating
    set of nodes of G (see algorithm 7 in [1]_) and computing the maximum
    flow between an arbitrary node in the dominating set and the rest of
    nodes in it. This is an implementation of algorithm 6 in [1]_. For
    directed graphs, the algorithm does n calls to the max flow function.
    The function raises an error if the directed graph is not weakly
    connected and returns an empty set if it is weakly connected.
    It is an implementation of algorithm 8 in [1]_.

    See also
    --------
    :meth:`minimum_st_edge_cut`
    :meth:`minimum_node_cut`
    :meth:`stoer_wagner`
    :meth:`node_connectivity`
    :meth:`edge_connectivity`
    :meth:`maximum_flow`
    :meth:`edmonds_karp`
    :meth:`preflow_push`
    :meth:`shortest_augmenting_path`

    References
    ----------
    .. [1] Abdol-Hossein Esfahanian. Connectivity Algorithms.
        http://www.cse.msu.edu/~cse835/Papers/Graph_connectivity_revised.pdf

    