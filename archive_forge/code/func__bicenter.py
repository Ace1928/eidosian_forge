import warnings
from collections import Counter, defaultdict
from math import comb, factorial
import networkx as nx
from networkx.utils import py_random_state
def _bicenter(n, cache, seed):
    """Returns a bi-centroidal tree on `n` nodes drawn uniformly at random.

    This function implements the algorithm Bicenter of [1]_.

    Parameters
    ----------
    n : int
        The number of nodes (must be even).
    cache : list of ints.
        Cache for :func:`_num_rooted_trees`.
    seed : random_state
        See :ref:`Randomness<randomness>`

    Returns
    -------
    (edges, n)
        The tree as a list of edges and number of nodes.

    References
    ----------
    .. [1] Wilf, Herbert S. "The uniform selection of free trees."
        Journal of Algorithms 2.2 (1981): 204-207.
        https://doi.org/10.1016/0196-6774(81)90021-3
    """
    t, t_nodes = _random_unlabeled_rooted_tree(n // 2, cache, seed)
    if seed.randint(0, _num_rooted_trees(n // 2, cache)) == 0:
        t2, t2_nodes = (t, t_nodes)
    else:
        t2, t2_nodes = _random_unlabeled_rooted_tree(n // 2, cache, seed)
    t.extend([(n1 + n // 2, n2 + n // 2) for n1, n2 in t2])
    t.append((0, n // 2))
    return (t, t_nodes + t2_nodes)