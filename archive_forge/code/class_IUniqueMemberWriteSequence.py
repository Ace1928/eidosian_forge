from zope.interface import Interface
from zope.interface.common import collections
class IUniqueMemberWriteSequence(Interface):
    """The write contract for a sequence that may enforce unique members"""

    def __setitem__(index, item):
        """``x.__setitem__(index, item) <==> x[index] = item``

        Declaring this interface does not specify whether `__setitem__`
        supports slice objects.
        """

    def __delitem__(index):
        """``x.__delitem__(index) <==> del x[index]``

        Declaring this interface does not specify whether `__delitem__`
        supports slice objects.
        """

    def __iadd__(y):
        """``x.__iadd__(y) <==> x += y``"""

    def append(item):
        """Append item to end"""

    def insert(index, item):
        """Insert item before index"""

    def pop(index=-1):
        """Remove and return item at index (default last)"""

    def remove(item):
        """Remove first occurrence of value"""

    def reverse():
        """Reverse *IN PLACE*"""

    def sort(cmpfunc=None):
        """Stable sort *IN PLACE*; `cmpfunc(x, y)` -> -1, 0, 1"""

    def extend(iterable):
        """Extend list by appending elements from the iterable"""