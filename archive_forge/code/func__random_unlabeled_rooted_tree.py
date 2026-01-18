import warnings
from collections import Counter, defaultdict
from math import comb, factorial
import networkx as nx
from networkx.utils import py_random_state
def _random_unlabeled_rooted_tree(n, cache_trees, seed):
    """Returns an unlabeled rooted tree with `n` nodes.

    Returns an unlabeled rooted tree with `n` nodes chosen uniformly
    at random using the "RANRUT" algorithm from [1]_.
    The tree is returned in the form: (list_of_edges, number_of_nodes)

    Parameters
    ----------
    n : int
        The number of nodes, greater than zero.
    cache_trees : list ints
        Cache for :func:`_num_rooted_trees`.
    seed : random_state
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    (list_of_edges, number_of_nodes) : list, int
        A random unlabeled rooted tree with `n` nodes as a 2-tuple
        ``(list_of_edges, number_of_nodes)``.
        The root is node 0.

    References
    ----------
    .. [1] Nijenhuis, Albert, and Wilf, Herbert S.
        "Combinatorial algorithms: for computers and calculators."
        Academic Press, 1978.
        https://doi.org/10.1016/C2013-0-11243-3
    """
    if n == 1:
        edges, n_nodes = ([], 1)
        return (edges, n_nodes)
    if n == 2:
        edges, n_nodes = ([(0, 1)], 2)
        return (edges, n_nodes)
    j, d = _select_jd_trees(n, cache_trees, seed)
    t1, t1_nodes = _random_unlabeled_rooted_tree(n - j * d, cache_trees, seed)
    t2, t2_nodes = _random_unlabeled_rooted_tree(d, cache_trees, seed)
    t12 = [(0, t2_nodes * i + t1_nodes) for i in range(j)]
    t1.extend(t12)
    for _ in range(j):
        t1.extend(((n1 + t1_nodes, n2 + t1_nodes) for n1, n2 in t2))
        t1_nodes += t2_nodes
    return (t1, t1_nodes)