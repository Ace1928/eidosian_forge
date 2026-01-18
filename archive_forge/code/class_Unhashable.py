import operator
import unittest
from cachetools import LRUCache, cachedmethod, keys
class Unhashable:

    def __init__(self, cache):
        self.cache = cache

    @cachedmethod(operator.attrgetter('cache'))
    def get_default(self, value):
        return value

    @cachedmethod(operator.attrgetter('cache'), key=keys.hashkey)
    def get_hashkey(self, value):
        return value

    def __hash__(self):
        raise TypeError('unhashable type')