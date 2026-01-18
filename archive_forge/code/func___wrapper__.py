import copy
import itertools
import operator
from functools import wraps
def __wrapper__(self, *args, __method_name=method_name, **kw):
    result = func(*self._args, **self._kw)
    return getattr(result, __method_name)(*args, **kw)