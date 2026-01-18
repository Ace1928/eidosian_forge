import networkx as nx
Returns the full join of graphs G and H.

    Full join is the union of G and H in which all edges between
    G and H are added.
    The node sets of G and H must be disjoint,
    otherwise an exception is raised.

    Parameters
    ----------
    G, H : graph
       A NetworkX graph

    rename : tuple , default=(None, None)
       Node names of G and H can be changed by specifying the tuple
       rename=('G-','H-') (for example).  Node "u" in G is then renamed
       "G-u" and "v" in H is renamed "H-v".

    Returns
    -------
    U : The full join graph with the same type as G.

    Notes
    -----
    It is recommended that G and H be either both directed or both undirected.

    If G is directed, then edges from G to H are added as well as from H to G.

    Note that full_join() does not produce parallel edges for MultiGraphs.

    The full join operation of graphs G and H is the same as getting
    their complement, performing a disjoint union, and finally getting
    the complement of the resulting graph.

    Graph, edge, and node attributes are propagated from G and H
    to the union graph.  If a graph attribute is present in both
    G and H the value from H is used.

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (0, 2)])
    >>> H = nx.Graph([(3, 4)])
    >>> R = nx.full_join(G, H, rename=("G", "H"))
    >>> R.nodes
    NodeView(('G0', 'G1', 'G2', 'H3', 'H4'))
    >>> R.edges
    EdgeView([('G0', 'G1'), ('G0', 'G2'), ('G0', 'H3'), ('G0', 'H4'), ('G1', 'H3'), ('G1', 'H4'), ('G2', 'H3'), ('G2', 'H4'), ('H3', 'H4')])

    See Also
    --------
    union
    disjoint_union
    