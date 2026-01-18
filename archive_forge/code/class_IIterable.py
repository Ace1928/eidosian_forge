import sys
from abc import ABCMeta
from collections import OrderedDict
from collections import UserDict
from collections import UserList
from collections import UserString
from collections import abc
from zope.interface.common import ABCInterface
from zope.interface.common import optional
class IIterable(ABCInterface):
    abc = abc.Iterable

    @optional
    def __iter__():
        """
        Optional method. If not provided, the interpreter will
        implement `iter` using the old ``__getitem__`` protocol.
        """