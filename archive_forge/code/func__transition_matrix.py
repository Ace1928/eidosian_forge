import networkx as nx
from networkx.utils import not_implemented_for
def _transition_matrix(G, nodelist=None, weight='weight', walk_type=None, alpha=0.95):
    """Returns the transition matrix of G.

    This is a row stochastic giving the transition probabilities while
    performing a random walk on the graph. Depending on the value of walk_type,
    P can be the transition matrix induced by a random walk, a lazy random walk,
    or a random walk with teleportation (PageRank).

    Parameters
    ----------
    G : DiGraph
       A NetworkX graph

    nodelist : list, optional
       The rows and columns are ordered according to the nodes in nodelist.
       If nodelist is None, then the ordering is produced by G.nodes().

    weight : string or None, optional (default='weight')
       The edge data key used to compute each value in the matrix.
       If None, then each edge has weight 1.

    walk_type : string or None, optional (default=None)
       If None, `P` is selected depending on the properties of the
       graph. Otherwise is one of 'random', 'lazy', or 'pagerank'

    alpha : real
       (1 - alpha) is the teleportation probability used with pagerank

    Returns
    -------
    P : numpy.ndarray
      transition matrix of G.

    Raises
    ------
    NetworkXError
        If walk_type not specified or alpha not in valid range
    """
    import numpy as np
    import scipy as sp
    if walk_type is None:
        if nx.is_strongly_connected(G):
            if nx.is_aperiodic(G):
                walk_type = 'random'
            else:
                walk_type = 'lazy'
        else:
            walk_type = 'pagerank'
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=weight, dtype=float)
    n, m = A.shape
    if walk_type in ['random', 'lazy']:
        DI = sp.sparse.csr_array(sp.sparse.spdiags(1.0 / A.sum(axis=1), 0, n, n))
        if walk_type == 'random':
            P = DI @ A
        else:
            I = sp.sparse.csr_array(sp.sparse.identity(n))
            P = (I + DI @ A) / 2.0
    elif walk_type == 'pagerank':
        if not 0 < alpha < 1:
            raise nx.NetworkXError('alpha must be between 0 and 1')
        A = A.toarray()
        A[A.sum(axis=1) == 0, :] = 1 / n
        A = A / A.sum(axis=1)[np.newaxis, :].T
        P = alpha * A + (1 - alpha) / n
    else:
        raise nx.NetworkXError('walk_type must be random, lazy, or pagerank')
    return P