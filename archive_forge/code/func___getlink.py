import collections
import collections.abc
import functools
import heapq
import random
import time
from .keys import hashkey as _defaultkey
def __getlink(self, key):
    value = self.__links[key]
    self.__links.move_to_end(key)
    return value