import sys
import warnings
from itertools import chain
from .sortedlist import SortedList, recursive_repr
from .sortedset import SortedSet
def __make_raise_attributeerror(original, alternate):
    message = 'SortedDict.{original}() is not implemented. Use SortedDict.{alternate}() instead.'.format(original=original, alternate=alternate)

    def method(self):
        raise AttributeError(message)
    method.__name__ = original
    method.__doc__ = message
    return property(method)