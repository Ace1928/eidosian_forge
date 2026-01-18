import collections
import collections.abc
import functools
import heapq
import random
import time
from .keys import hashkey as _defaultkey
def _methodkey(_, *args, **kwargs):
    return _defaultkey(*args, **kwargs)