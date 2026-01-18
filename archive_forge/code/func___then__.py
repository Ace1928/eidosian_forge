import operator
import sys
from functools import reduce
from importlib import import_module
from types import ModuleType
def __then__(self, fun, *args, **kwargs):
    if self.__evaluated__():
        return fun(*args, **kwargs)
    from collections import deque
    try:
        pending = object.__getattribute__(self, '__pending__')
    except AttributeError:
        pending = None
    if pending is None:
        pending = deque()
        object.__setattr__(self, '__pending__', pending)
    pending.append((fun, args, kwargs))