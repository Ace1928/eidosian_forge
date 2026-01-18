import collections.abc
class OrderPreservingSet(collections.abc.MutableSet):
    """A set based on dict so that it preserves key insertion order."""

    def __init__(self, iterable=None):
        self._dict = {item: None for item in iterable or []}

    def __len__(self):
        return len(self._dict)

    def __contains__(self, value):
        return value in self._dict

    def __iter__(self):
        return iter(self._dict)

    def add(self, item):
        self._dict[item] = None

    def discard(self, value):
        del self._dict[value]

    def clear(self):
        self._dict = {}

    def __eq__(self, other):
        if not isinstance(other, OrderPreservingSet):
            return NotImplemented
        return self._dict.keys() == other._dict.keys()

    def __le__(self, other):
        if not isinstance(other, OrderPreservingSet):
            return NotImplemented
        return self._dict.keys() <= other._dict.keys()

    def __ge__(self, other):
        if not isinstance(other, OrderPreservingSet):
            return NotImplemented
        return self._dict.keys() >= other._dict.keys()

    def __and__(self, other):
        return self._from_iterable((value for value in self if value in other))

    def __or__(self, other):
        if not isinstance(other, OrderPreservingSet):
            raise TypeError("cannot union an 'OrderPreservingSet' with an unordered iterable.")
        result = self._from_iterable((value for value in self))
        for value in other:
            result._dict[value] = None
        return result

    def union(self, other):
        return self | other