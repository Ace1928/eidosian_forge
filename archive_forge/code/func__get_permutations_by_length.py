import itertools
from collections import Counter, defaultdict
from functools import reduce, wraps
@staticmethod
def _get_permutations_by_length(items):
    """
        Get all permutations of items, but only permute items with the same
        length.

        >>> found = list(ISMAGS._get_permutations_by_length([[1], [2], [3, 4], [4, 5]]))
        >>> answer = [
        ...     (([1], [2]), ([3, 4], [4, 5])),
        ...     (([1], [2]), ([4, 5], [3, 4])),
        ...     (([2], [1]), ([3, 4], [4, 5])),
        ...     (([2], [1]), ([4, 5], [3, 4])),
        ... ]
        >>> found == answer
        True
        """
    by_len = defaultdict(list)
    for item in items:
        by_len[len(item)].append(item)
    yield from itertools.product(*(itertools.permutations(by_len[l]) for l in sorted(by_len)))