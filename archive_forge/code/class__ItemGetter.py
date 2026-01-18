from __future__ import annotations
from sympy.core import Symbol
from sympy.utilities.iterables import iterable
class _ItemGetter:
    """Helper class to return a subsequence of values."""

    def __init__(self, seq):
        self.seq = tuple(seq)

    def __call__(self, m):
        return tuple((m[idx] for idx in self.seq))

    def __eq__(self, other):
        if not isinstance(other, _ItemGetter):
            return False
        return self.seq == other.seq