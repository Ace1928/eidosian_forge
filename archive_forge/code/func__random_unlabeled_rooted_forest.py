import warnings
from collections import Counter, defaultdict
from math import comb, factorial
import networkx as nx
from networkx.utils import py_random_state
def _random_unlabeled_rooted_forest(n, q, cache_trees, cache_forests, seed):
    """Returns an unlabeled rooted forest with `n` nodes, and with no more
    than `q` nodes per tree, drawn uniformly at random. It is an implementation
    of the algorithm "Forest" of [1]_.

    Parameters
    ----------
    n : int
        The number of nodes.
    q : int
        The maximum number of nodes per tree.
    cache_trees :
        Cache for :func:`_num_rooted_trees`.
    cache_forests :
        Cache for :func:`_num_rooted_forests`.
    seed : random_state
       See :ref:`Randomness<randomness>`.

    Returns
    -------
    (edges, n, r) : (list, int, list)
        The forest (edges, n) and a list r of root nodes.

    References
    ----------
    .. [1] Wilf, Herbert S. "The uniform selection of free trees."
        Journal of Algorithms 2.2 (1981): 204-207.
        https://doi.org/10.1016/0196-6774(81)90021-3
    """
    if n == 0:
        return ([], 0, [])
    j, d = _select_jd_forests(n, q, cache_forests, seed)
    t1, t1_nodes, r1 = _random_unlabeled_rooted_forest(n - j * d, q, cache_trees, cache_forests, seed)
    t2, t2_nodes = _random_unlabeled_rooted_tree(d, cache_trees, seed)
    for _ in range(j):
        r1.append(t1_nodes)
        t1.extend(((n1 + t1_nodes, n2 + t1_nodes) for n1, n2 in t2))
        t1_nodes += t2_nodes
    return (t1, t1_nodes, r1)