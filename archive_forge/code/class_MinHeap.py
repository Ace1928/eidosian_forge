from heapq import heappop, heappush
from itertools import count
import networkx as nx
class MinHeap:
    """Base class for min-heaps.

    A MinHeap stores a collection of key-value pairs ordered by their values.
    It supports querying the minimum pair, inserting a new pair, decreasing the
    value in an existing pair and deleting the minimum pair.
    """

    class _Item:
        """Used by subclassess to represent a key-value pair."""
        __slots__ = ('key', 'value')

        def __init__(self, key, value):
            self.key = key
            self.value = value

        def __repr__(self):
            return repr((self.key, self.value))

    def __init__(self):
        """Initialize a new min-heap."""
        self._dict = {}

    def min(self):
        """Query the minimum key-value pair.

        Returns
        -------
        key, value : tuple
            The key-value pair with the minimum value in the heap.

        Raises
        ------
        NetworkXError
            If the heap is empty.
        """
        raise NotImplementedError

    def pop(self):
        """Delete the minimum pair in the heap.

        Returns
        -------
        key, value : tuple
            The key-value pair with the minimum value in the heap.

        Raises
        ------
        NetworkXError
            If the heap is empty.
        """
        raise NotImplementedError

    def get(self, key, default=None):
        """Returns the value associated with a key.

        Parameters
        ----------
        key : hashable object
            The key to be looked up.

        default : object
            Default value to return if the key is not present in the heap.
            Default value: None.

        Returns
        -------
        value : object.
            The value associated with the key.
        """
        raise NotImplementedError

    def insert(self, key, value, allow_increase=False):
        """Insert a new key-value pair or modify the value in an existing
        pair.

        Parameters
        ----------
        key : hashable object
            The key.

        value : object comparable with existing values.
            The value.

        allow_increase : bool
            Whether the value is allowed to increase. If False, attempts to
            increase an existing value have no effect. Default value: False.

        Returns
        -------
        decreased : bool
            True if a pair is inserted or the existing value is decreased.
        """
        raise NotImplementedError

    def __nonzero__(self):
        """Returns whether the heap if empty."""
        return bool(self._dict)

    def __bool__(self):
        """Returns whether the heap if empty."""
        return bool(self._dict)

    def __len__(self):
        """Returns the number of key-value pairs in the heap."""
        return len(self._dict)

    def __contains__(self, key):
        """Returns whether a key exists in the heap.

        Parameters
        ----------
        key : any hashable object.
            The key to be looked up.
        """
        return key in self._dict