import sys
from collections.abc import MutableSet
from .compat import queue
class SkipRepeatsQueue(queue.Queue):
    """Thread-safe implementation of an special queue where a
    put of the last-item put'd will be dropped.

    The implementation leverages locking already implemented in the base class
    redefining only the primitives.

    Queued items must be immutable and hashable so that they can be used
    as dictionary keys. You must implement **only read-only properties** and
    the :meth:`Item.__hash__()`, :meth:`Item.__eq__()`, and
    :meth:`Item.__ne__()` methods for items to be hashable.

    An example implementation follows::

        class Item(object):
            def __init__(self, a, b):
                self._a = a
                self._b = b

            @property
            def a(self):
                return self._a

            @property
            def b(self):
                return self._b

            def _key(self):
                return (self._a, self._b)

            def __eq__(self, item):
                return self._key() == item._key()

            def __ne__(self, item):
                return self._key() != item._key()

            def __hash__(self):
                return hash(self._key())

    based on the OrderedSetQueue below
    """

    def _init(self, maxsize):
        queue.Queue._init(self, maxsize)
        self._last_item = None

    def _put(self, item):
        if item != self._last_item:
            queue.Queue._put(self, item)
            self._last_item = item
        else:
            self.unfinished_tasks -= 1

    def _get(self):
        item = queue.Queue._get(self)
        if item is self._last_item:
            self._last_item = None
        return item