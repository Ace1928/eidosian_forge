import sys
from abc import ABCMeta
from collections import OrderedDict
from collections import UserDict
from collections import UserList
from collections import UserString
from collections import abc
from zope.interface.common import ABCInterface
from zope.interface.common import optional
class IReversible(IIterable):
    abc = _new_in_ver('Reversible', True, (IIterable.getABC(),))

    @optional
    def __reversed__():
        """
        Optional method. If this isn't present, the interpreter
        will use ``__len__`` and ``__getitem__`` to implement the
        `reversed` builtin.
        """