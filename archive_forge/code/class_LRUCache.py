import re
from six.moves import html_entities as entities
import six
class LRUCache(dict):
    """A dictionary-like object that stores only a certain number of items, and
    discards its least recently used item when full.
    
    >>> cache = LRUCache(3)
    >>> cache['A'] = 0
    >>> cache['B'] = 1
    >>> cache['C'] = 2
    >>> len(cache)
    3
    
    >>> cache['A']
    0
    
    Adding new items to the cache does not increase its size. Instead, the least
    recently used item is dropped:
    
    >>> cache['D'] = 3
    >>> len(cache)
    3
    >>> 'B' in cache
    False
    
    Iterating over the cache returns the keys, starting with the most recently
    used:
    
    >>> for key in cache:
    ...     print(key)
    D
    A
    C

    This code is based on the LRUCache class from ``myghtyutils.util``, written
    by Mike Bayer and released under the MIT license. See:

      http://svn.myghty.org/myghtyutils/trunk/lib/myghtyutils/util.py
    """

    class _Item(object):

        def __init__(self, key, value):
            self.prv = self.nxt = None
            self.key = key
            self.value = value

        def __repr__(self):
            return repr(self.value)

    def __init__(self, capacity):
        self._dict = dict()
        self.capacity = capacity
        self.head = None
        self.tail = None

    def __contains__(self, key):
        return key in self._dict

    def __iter__(self):
        cur = self.head
        while cur:
            yield cur.key
            cur = cur.nxt

    def __len__(self):
        return len(self._dict)

    def __getitem__(self, key):
        item = self._dict[key]
        self._update_item(item)
        return item.value

    def __setitem__(self, key, value):
        item = self._dict.get(key)
        if item is None:
            item = self._Item(key, value)
            self._dict[key] = item
            self._insert_item(item)
        else:
            item.value = value
            self._update_item(item)
            self._manage_size()

    def __repr__(self):
        return repr(self._dict)

    def _insert_item(self, item):
        item.prv = None
        item.nxt = self.head
        if self.head is not None:
            self.head.prv = item
        else:
            self.tail = item
        self.head = item
        self._manage_size()

    def _manage_size(self):
        while len(self._dict) > self.capacity:
            del self._dict[self.tail.key]
            if self.tail != self.head:
                self.tail = self.tail.prv
                self.tail.nxt = None
            else:
                self.head = self.tail = None

    def _update_item(self, item):
        if self.head == item:
            return
        prv = item.prv
        prv.nxt = item.nxt
        if item.nxt is not None:
            item.nxt.prv = prv
        else:
            self.tail = prv
        item.prv = None
        item.nxt = self.head
        self.head.prv = self.head = item