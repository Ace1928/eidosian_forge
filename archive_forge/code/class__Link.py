import collections
import collections.abc
import functools
import heapq
import random
import time
from .keys import hashkey as _defaultkey
class _Link:
    __slots__ = ('key', 'expires', 'next', 'prev')

    def __init__(self, key=None, expires=None):
        self.key = key
        self.expires = expires

    def __reduce__(self):
        return (TTLCache._Link, (self.key, self.expires))

    def unlink(self):
        next = self.next
        prev = self.prev
        prev.next = next
        next.prev = prev