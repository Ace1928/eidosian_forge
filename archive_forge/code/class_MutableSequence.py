from abc import ABCMeta, abstractmethod
import sys
class MutableSequence(Sequence):
    """All the operations on a read-write sequence.

    Concrete subclasses must provide __new__ or __init__,
    __getitem__, __setitem__, __delitem__, __len__, and insert().
    """
    __slots__ = ()

    @abstractmethod
    def __setitem__(self, index, value):
        raise IndexError

    @abstractmethod
    def __delitem__(self, index):
        raise IndexError

    @abstractmethod
    def insert(self, index, value):
        """S.insert(index, value) -- insert value before index"""
        raise IndexError

    def append(self, value):
        """S.append(value) -- append value to the end of the sequence"""
        self.insert(len(self), value)

    def clear(self):
        """S.clear() -> None -- remove all items from S"""
        try:
            while True:
                self.pop()
        except IndexError:
            pass

    def reverse(self):
        """S.reverse() -- reverse *IN PLACE*"""
        n = len(self)
        for i in range(n // 2):
            self[i], self[n - i - 1] = (self[n - i - 1], self[i])

    def extend(self, values):
        """S.extend(iterable) -- extend sequence by appending elements from the iterable"""
        if values is self:
            values = list(values)
        for v in values:
            self.append(v)

    def pop(self, index=-1):
        """S.pop([index]) -> item -- remove and return item at index (default last).
           Raise IndexError if list is empty or index is out of range.
        """
        v = self[index]
        del self[index]
        return v

    def remove(self, value):
        """S.remove(value) -- remove first occurrence of value.
           Raise ValueError if the value is not present.
        """
        del self[self.index(value)]

    def __iadd__(self, values):
        self.extend(values)
        return self