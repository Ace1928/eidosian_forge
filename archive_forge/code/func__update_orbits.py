import itertools
from collections import Counter, defaultdict
from functools import reduce, wraps
@staticmethod
def _update_orbits(orbits, permutations):
    """
        Update orbits based on permutations. Orbits is modified in place.
        For every pair of items in permutations their respective orbits are
        merged.
        """
    for permutation in permutations:
        node, node2 = permutation
        first = second = None
        for idx, orbit in enumerate(orbits):
            if first is not None and second is not None:
                break
            if node in orbit:
                first = idx
            if node2 in orbit:
                second = idx
        if first != second:
            orbits[first].update(orbits[second])
            del orbits[second]