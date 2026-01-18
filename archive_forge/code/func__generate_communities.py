import itertools
import math
import networkx as nx
from networkx.utils import py_random_state
def _generate_communities(degree_seq, community_sizes, mu, max_iters, seed):
    """Returns a list of sets, each of which represents a community.

    ``degree_seq`` is the degree sequence that must be met by the
    graph.

    ``community_sizes`` is the community size distribution that must be
    met by the generated list of sets.

    ``mu`` is a float in the interval [0, 1] indicating the fraction of
    intra-community edges incident to each node.

    ``max_iters`` is the number of times to try to add a node to a
    community. This must be greater than the length of
    ``degree_seq``, otherwise this function will always fail. If
    the number of iterations exceeds this value,
    :exc:`~networkx.exception.ExceededMaxIterations` is raised.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    The communities returned by this are sets of integers in the set {0,
    ..., *n* - 1}, where *n* is the length of ``degree_seq``.

    """
    result = [set() for _ in community_sizes]
    n = len(degree_seq)
    free = list(range(n))
    for i in range(max_iters):
        v = free.pop()
        c = seed.choice(range(len(community_sizes)))
        s = round(degree_seq[v] * (1 - mu))
        if s < community_sizes[c]:
            result[c].add(v)
        else:
            free.append(v)
        if len(result[c]) > community_sizes[c]:
            free.append(result[c].pop())
        if not free:
            return result
    msg = 'Could not assign communities; try increasing min_community'
    raise nx.ExceededMaxIterations(msg)