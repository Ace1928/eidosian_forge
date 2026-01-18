import os
import atexit
import functools
import pickle
import sys
import time
import warnings
import numpy as np
class DummyMPI:
    rank = 0
    size = 1

    def _returnval(self, a, root=-1):
        if np.isscalar(a):
            return a
        if hasattr(a, '__array__'):
            a = a.__array__()
        assert isinstance(a, np.ndarray)
        return None

    def sum(self, a, root=-1):
        return self._returnval(a)

    def product(self, a, root=-1):
        return self._returnval(a)

    def broadcast(self, a, root):
        assert root == 0
        return self._returnval(a)

    def barrier(self):
        pass