import itertools
from collections import Counter, defaultdict
from functools import reduce, wraps
@staticmethod
def _remove_node(node, nodes, constraints):
    """
        Returns a new set where node has been removed from nodes, subject to
        symmetry constraints. We know, that for every constraint we have
        those subgraph nodes are equal. So whenever we would remove the
        lower part of a constraint, remove the higher instead.
        """
    while True:
        for low, high in constraints:
            if low == node and high in nodes:
                node = high
                break
        else:
            break
    return frozenset(nodes - {node})