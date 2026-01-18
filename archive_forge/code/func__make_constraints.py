import itertools
from collections import Counter, defaultdict
from functools import reduce, wraps
@staticmethod
def _make_constraints(cosets):
    """
        Turn cosets into constraints.
        """
    constraints = []
    for node_i, node_ts in cosets.items():
        for node_t in node_ts:
            if node_i != node_t:
                constraints.append((node_i, node_t))
    return constraints