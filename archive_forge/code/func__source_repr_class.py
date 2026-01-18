import copy
import types
from itertools import count
def _source_repr_class(self, source, binding=None):
    d = self.__dict__.copy()
    if 'declarative_count' in d:
        del d['declarative_count']
    return source.makeClass(self, binding, d, (self.__class__,))