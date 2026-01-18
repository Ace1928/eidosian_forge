import networkx as nx
def _hits_numpy(G, normalized=True):
    """Returns HITS hubs and authorities values for nodes.

    The HITS algorithm computes two numbers for a node.
    Authorities estimates the node value based on the incoming links.
    Hubs estimates the node value based on outgoing links.

    Parameters
    ----------
    G : graph
      A NetworkX graph

    normalized : bool (default=True)
       Normalize results by the sum of all of the values.

    Returns
    -------
    (hubs,authorities) : two-tuple of dictionaries
       Two dictionaries keyed by node containing the hub and authority
       values.

    Examples
    --------
    >>> G = nx.path_graph(4)

    The `hubs` and `authorities` are given by the eigenvectors corresponding to the
    maximum eigenvalues of the hubs_matrix and the authority_matrix, respectively.

    The ``hubs`` and ``authority`` matrices are computed from the adjacency
    matrix:

    >>> adj_ary = nx.to_numpy_array(G)
    >>> hubs_matrix = adj_ary @ adj_ary.T
    >>> authority_matrix = adj_ary.T @ adj_ary

    `_hits_numpy` maps the eigenvector corresponding to the maximum eigenvalue
    of the respective matrices to the nodes in `G`:

    >>> from networkx.algorithms.link_analysis.hits_alg import _hits_numpy
    >>> hubs, authority = _hits_numpy(G)

    Notes
    -----
    The eigenvector calculation uses NumPy's interface to LAPACK.

    The HITS algorithm was designed for directed graphs but this
    algorithm does not check if the input graph is directed and will
    execute on undirected graphs.

    References
    ----------
    .. [1] A. Langville and C. Meyer,
       "A survey of eigenvector methods of web information retrieval."
       http://citeseer.ist.psu.edu/713792.html
    .. [2] Jon Kleinberg,
       Authoritative sources in a hyperlinked environment
       Journal of the ACM 46 (5): 604-32, 1999.
       doi:10.1145/324133.324140.
       http://www.cs.cornell.edu/home/kleinber/auth.pdf.
    """
    import numpy as np
    if len(G) == 0:
        return ({}, {})
    adj_ary = nx.to_numpy_array(G)
    H = adj_ary @ adj_ary.T
    e, ev = np.linalg.eig(H)
    h = ev[:, np.argmax(e)]
    A = adj_ary.T @ adj_ary
    e, ev = np.linalg.eig(A)
    a = ev[:, np.argmax(e)]
    if normalized:
        h /= h.sum()
        a /= a.sum()
    else:
        h /= h.max()
        a /= a.max()
    hubs = dict(zip(G, map(float, h)))
    authorities = dict(zip(G, map(float, a)))
    return (hubs, authorities)