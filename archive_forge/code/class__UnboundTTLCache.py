import collections
import functools
import math
import random
import time
from . import FIFOCache, LFUCache, LRUCache, MRUCache, RRCache, TTLCache
from . import keys
class _UnboundTTLCache(TTLCache):

    def __init__(self, ttl, timer):
        TTLCache.__init__(self, math.inf, ttl, timer)

    @property
    def maxsize(self):
        return None