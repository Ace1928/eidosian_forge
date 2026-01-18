import math
import networkx as nx
from networkx.utils import not_implemented_for
Compute the eigenvector centrality for the graph G.

    Eigenvector centrality computes the centrality for a node by adding
    the centrality of its predecessors. The centrality for node $i$ is the
    $i$-th element of a left eigenvector associated with the eigenvalue $\lambda$
    of maximum modulus that is positive. Such an eigenvector $x$ is
    defined up to a multiplicative constant by the equation

    .. math::

         \lambda x^T = x^T A,

    where $A$ is the adjacency matrix of the graph G. By definition of
    row-column product, the equation above is equivalent to

    .. math::

        \lambda x_i = \sum_{j\to i}x_j.

    That is, adding the eigenvector centralities of the predecessors of
    $i$ one obtains the eigenvector centrality of $i$ multiplied by
    $\lambda$. In the case of undirected graphs, $x$ also solves the familiar
    right-eigenvector equation $Ax = \lambda x$.

    By virtue of the Perron–Frobenius theorem [1]_, if G is strongly
    connected there is a unique eigenvector $x$, and all its entries
    are strictly positive.

    If G is not strongly connected there might be several left
    eigenvectors associated with $\lambda$, and some of their elements
    might be zero.

    Parameters
    ----------
    G : graph
      A networkx graph.

    max_iter : integer, optional (default=50)
      Maximum number of Arnoldi update iterations allowed.

    tol : float, optional (default=0)
      Relative accuracy for eigenvalues (stopping criterion).
      The default value of 0 implies machine precision.

    weight : None or string, optional (default=None)
      If None, all edge weights are considered equal. Otherwise holds the
      name of the edge attribute used as weight. In this measure the
      weight is interpreted as the connection strength.

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with eigenvector centrality as the value. The
       associated vector has unit Euclidean norm and the values are
       nonegative.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> centrality = nx.eigenvector_centrality_numpy(G)
    >>> print([f"{node} {centrality[node]:0.2f}" for node in centrality])
    ['0 0.37', '1 0.60', '2 0.60', '3 0.37']

    Raises
    ------
    NetworkXPointlessConcept
        If the graph G is the null graph.

    ArpackNoConvergence
        When the requested convergence is not obtained. The currently
        converged eigenvalues and eigenvectors can be found as
        eigenvalues and eigenvectors attributes of the exception object.

    See Also
    --------
    :func:`scipy.sparse.linalg.eigs`
    eigenvector_centrality
    :func:`~networkx.algorithms.link_analysis.pagerank_alg.pagerank`
    :func:`~networkx.algorithms.link_analysis.hits_alg.hits`

    Notes
    -----
    Eigenvector centrality was introduced by Landau [2]_ for chess
    tournaments. It was later rediscovered by Wei [3]_ and then
    popularized by Kendall [4]_ in the context of sport ranking. Berge
    introduced a general definition for graphs based on social connections
    [5]_. Bonacich [6]_ reintroduced again eigenvector centrality and made
    it popular in link analysis.

    This function computes the left dominant eigenvector, which corresponds
    to adding the centrality of predecessors: this is the usual approach.
    To add the centrality of successors first reverse the graph with
    ``G.reverse()``.

    This implementation uses the
    :func:`SciPy sparse eigenvalue solver<scipy.sparse.linalg.eigs>` (ARPACK)
    to find the largest eigenvalue/eigenvector pair using Arnoldi iterations
    [7]_.

    References
    ----------
    .. [1] Abraham Berman and Robert J. Plemmons.
       "Nonnegative Matrices in the Mathematical Sciences."
       Classics in Applied Mathematics. SIAM, 1994.

    .. [2] Edmund Landau.
       "Zur relativen Wertbemessung der Turnierresultate."
       Deutsches Wochenschach, 11:366–369, 1895.

    .. [3] Teh-Hsing Wei.
       "The Algebraic Foundations of Ranking Theory."
       PhD thesis, University of Cambridge, 1952.

    .. [4] Maurice G. Kendall.
       "Further contributions to the theory of paired comparisons."
       Biometrics, 11(1):43–62, 1955.
       https://www.jstor.org/stable/3001479

    .. [5] Claude Berge
       "Théorie des graphes et ses applications."
       Dunod, Paris, France, 1958.

    .. [6] Phillip Bonacich.
       "Technique for analyzing overlapping memberships."
       Sociological Methodology, 4:176–185, 1972.
       https://www.jstor.org/stable/270732

    .. [7] Arnoldi iteration:: https://en.wikipedia.org/wiki/Arnoldi_iteration

    