import sys
from abc import ABCMeta
from collections import OrderedDict
from collections import UserDict
from collections import UserList
from collections import UserString
from collections import abc
from zope.interface.common import ABCInterface
from zope.interface.common import optional
class IValuesView(IMappingView, ICollection):
    abc = abc.ValuesView

    @optional
    def __contains__(other):
        """
        Optional method. If not provided, the interpreter will use
        ``__iter__`` or the old ``__len__`` and ``__getitem__`` protocol
        to implement ``in``.
        """