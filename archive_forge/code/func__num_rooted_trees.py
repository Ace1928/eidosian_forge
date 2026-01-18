import warnings
from collections import Counter, defaultdict
from math import comb, factorial
import networkx as nx
from networkx.utils import py_random_state
def _num_rooted_trees(n, cache_trees):
    """Returns the number of unlabeled rooted trees with `n` nodes.

    See also https://oeis.org/A000081.

    Parameters
    ----------
    n : int
        The number of nodes
    cache_trees : list of ints
        The $i$-th element is the number of unlabeled rooted trees with $i$ nodes,
        which is used as a cache (and is extended to length $n+1$ if needed)

    Returns
    -------
    int
        The number of unlabeled rooted trees with `n` nodes.
    """
    for n_i in range(len(cache_trees), n + 1):
        cache_trees.append(sum([d * cache_trees[n_i - j * d] * cache_trees[d] for d in range(1, n_i) for j in range(1, (n_i - 1) // d + 1)]) // (n_i - 1))
    return cache_trees[n]