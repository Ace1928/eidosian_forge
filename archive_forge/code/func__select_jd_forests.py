import warnings
from collections import Counter, defaultdict
from math import comb, factorial
import networkx as nx
from networkx.utils import py_random_state
def _select_jd_forests(n, q, cache_forests, seed):
    """Given `n` and `q`, returns a pair of positive integers $(j,d)$
    such that $j\\leq d$, with probability satisfying (F1) of [1]_.

    Parameters
    ----------
    n : int
        The number of nodes.
    q : int
        The maximum number of nodes for each tree of the forest.
    cache_forests : list of ints
        Cache for :func:`_num_rooted_forests`.
    seed : random_state
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    (int, int)
        A pair of positive integers $(j,d)$

    References
    ----------
    .. [1] Wilf, Herbert S. "The uniform selection of free trees."
        Journal of Algorithms 2.2 (1981): 204-207.
        https://doi.org/10.1016/0196-6774(81)90021-3
    """
    p = seed.randint(0, _num_rooted_forests(n, q, cache_forests) * n - 1)
    cumsum = 0
    for d in range(q, 0, -1):
        for j in range(1, n // d + 1):
            cumsum += d * _num_rooted_forests(n - j * d, q, cache_forests) * _num_rooted_forests(d - 1, q, cache_forests)
            if p < cumsum:
                return (j, d)