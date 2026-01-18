import copy
from collections import OrderedDict
from contextlib import contextmanager
from threading import RLock
from typing import Optional
class WrappedObj:
    """Wraps an object to make its hash dependent on its identity"""

    def __init__(self, obj):
        self.obj = obj

    def __hash__(self):
        return id(self.obj)

    def __eq__(self, other):
        if not isinstance(other, WrappedObj):
            return False
        return id(self.obj) == id(other.obj)

    def __repr__(self):
        return f'Wrapped({self.obj.__repr__()})'