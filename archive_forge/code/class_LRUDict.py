import sys
import copy
import heapq
import collections
import functools
import numpy as np
from scipy._lib._util import MapWrapper, _FunctionWrapper
class LRUDict(collections.OrderedDict):

    def __init__(self, max_size):
        self.__max_size = max_size

    def __setitem__(self, key, value):
        existing_key = key in self
        super().__setitem__(key, value)
        if existing_key:
            self.move_to_end(key)
        elif len(self) > self.__max_size:
            self.popitem(last=False)

    def update(self, other):
        raise NotImplementedError()