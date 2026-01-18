from itertools import chain, repeat
import networkx as nx
Returns a new graph that contains only the nodes and the edges that exist in
    all graphs.

    Parameters
    ----------
    graphs : iterable
       Iterable of NetworkX graphs

    Returns
    -------
    R : A new graph with the same type as the first graph in list

    Raises
    ------
    ValueError
       If `graphs` is an empty list.

    NetworkXError
        In case of mixed type graphs, like MultiGraph and Graph, or directed and undirected graphs.

    Notes
    -----
    For operating on mixed type graphs, they should be converted to the same type.

    Attributes from the graph, nodes, and edges are not copied to the new
    graph.

    The resulting graph can be updated with attributes if desired. For example, code which adds the minimum attribute for each node across all graphs could work.
    >>> g = nx.Graph()
    >>> g.add_node(0, capacity=4)
    >>> g.add_node(1, capacity=3)
    >>> g.add_edge(0, 1)

    >>> h = g.copy()
    >>> h.nodes[0]["capacity"] = 2

    >>> gh = nx.intersection_all([g, h])

    >>> new_node_attr = {n: min(*(anyG.nodes[n].get('capacity', float('inf')) for anyG in [g, h])) for n in gh}
    >>> nx.set_node_attributes(gh, new_node_attr, 'new_capacity')
    >>> gh.nodes(data=True)
    NodeDataView({0: {'new_capacity': 2}, 1: {'new_capacity': 3}})

    Examples
    --------
    >>> G1 = nx.Graph([(1, 2), (2, 3)])
    >>> G2 = nx.Graph([(2, 3), (3, 4)])
    >>> R = nx.intersection_all([G1, G2])
    >>> list(R.nodes())
    [2, 3]
    >>> list(R.edges())
    [(2, 3)]

    