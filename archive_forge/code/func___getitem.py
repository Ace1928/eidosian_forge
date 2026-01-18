import collections
import collections.abc
import functools
import heapq
import random
import time
from .keys import hashkey as _defaultkey
def __getitem(self, key):
    value = self.__items[key]
    self.__items.move_to_end(key)
    return value