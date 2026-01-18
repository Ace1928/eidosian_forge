import collections
import collections.abc
import functools
import heapq
import random
import time
from .keys import hashkey as _defaultkey
class _DefaultSize:
    __slots__ = ()

    def __getitem__(self, _):
        return 1

    def __setitem__(self, _, value):
        assert value == 1

    def pop(self, _):
        return 1