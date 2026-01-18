import warnings
from collections import Counter, defaultdict
from math import comb, factorial
import networkx as nx
from networkx.utils import py_random_state
def _num_trees(n, cache_trees):
    """Returns the number of unlabeled trees with `n` nodes.

    See also https://oeis.org/A000055.

    Parameters
    ----------
    n : int
        The number of nodes.
    cache_trees : list of ints
        Cache for :func:`_num_rooted_trees`.

    Returns
    -------
    int
        The number of unlabeled trees with `n` nodes.
    """
    r = _num_rooted_trees(n, cache_trees) - sum([_num_rooted_trees(j, cache_trees) * _num_rooted_trees(n - j, cache_trees) for j in range(1, n // 2 + 1)])
    if n % 2 == 0:
        r += comb(_num_rooted_trees(n // 2, cache_trees) + 1, 2)
    return r