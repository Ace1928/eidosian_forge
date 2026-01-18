from zope.interface import Interface
from zope.interface.common import collections
class IExtendedReadMapping(IIterableMapping):
    """
    Something with a particular method equivalent to ``__contains__``.

    On Python 2, `dict` provided the ``has_key`` method, but it was removed
    in Python 3.
    """