import sys
import copy
import heapq
import collections
import functools
import numpy as np
from scipy._lib._util import MapWrapper, _FunctionWrapper
def _get_sizeof(obj):
    try:
        return sys.getsizeof(obj)
    except TypeError:
        if hasattr(obj, '__sizeof__'):
            return int(obj.__sizeof__())
        return 64