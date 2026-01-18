import copy
import types
from itertools import count
@classmethod
def __classsourcerepr__(cls, source, binding=None):
    d = cls.__dict__.copy()
    del d['declarative_count']
    return source.makeClass(cls, binding or cls.__name__, d, cls.__bases__)