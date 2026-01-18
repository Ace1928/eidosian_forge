from math import log
import networkx as nx
from networkx.utils import not_implemented_for
@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch
def adamic_adar_index(G, ebunch=None):
    """Compute the Adamic-Adar index of all node pairs in ebunch.

    Adamic-Adar index of `u` and `v` is defined as

    .. math::

        \\sum_{w \\in \\Gamma(u) \\cap \\Gamma(v)} \\frac{1}{\\log |\\Gamma(w)|}

    where $\\Gamma(u)$ denotes the set of neighbors of $u$.
    This index leads to zero-division for nodes only connected via self-loops.
    It is intended to be used when no self-loops are present.

    Parameters
    ----------
    G : graph
        NetworkX undirected graph.

    ebunch : iterable of node pairs, optional (default = None)
        Adamic-Adar index will be computed for each pair of nodes given
        in the iterable. The pairs must be given as 2-tuples (u, v)
        where u and v are nodes in the graph. If ebunch is None then all
        nonexistent edges in the graph will be used.
        Default value: None.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their Adamic-Adar index.

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> preds = nx.adamic_adar_index(G, [(0, 1), (2, 3)])
    >>> for u, v, p in preds:
    ...     print(f"({u}, {v}) -> {p:.8f}")
    (0, 1) -> 2.16404256
    (2, 3) -> 2.16404256

    References
    ----------
    .. [1] D. Liben-Nowell, J. Kleinberg.
           The Link Prediction Problem for Social Networks (2004).
           http://www.cs.cornell.edu/home/kleinber/link-pred.pdf
    """

    def predict(u, v):
        return sum((1 / log(G.degree(w)) for w in nx.common_neighbors(G, u, v)))
    return _apply_prediction(G, predict, ebunch)