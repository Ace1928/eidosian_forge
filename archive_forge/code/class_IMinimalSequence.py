from zope.interface import Interface
from zope.interface.common import collections
class IMinimalSequence(collections.IIterable):
    """Most basic sequence interface.

    All sequences are iterable.  This requires at least one of the
    following:

    - a `__getitem__()` method that takes a single argument; integer
      values starting at 0 must be supported, and `IndexError` should
      be raised for the first index for which there is no value, or

    - an `__iter__()` method that returns an iterator as defined in
      the Python documentation (http://docs.python.org/lib/typeiter.html).

    """

    def __getitem__(index):
        """``x.__getitem__(index) <==> x[index]``

        Declaring this interface does not specify whether `__getitem__`
        supports slice objects."""