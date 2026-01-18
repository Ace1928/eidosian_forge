import sys
import warnings
from itertools import chain
from .sortedlist import SortedList, recursive_repr
from .sortedset import SortedSet
class _NotGiven(object):

    def __repr__(self):
        return '<not-given>'