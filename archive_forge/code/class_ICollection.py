import sys
from abc import ABCMeta
from collections import OrderedDict
from collections import UserDict
from collections import UserList
from collections import UserString
from collections import abc
from zope.interface.common import ABCInterface
from zope.interface.common import optional
class ICollection(ISized, IIterable, IContainer):
    abc = _new_in_ver('Collection', True, (ISized.getABC(), IIterable.getABC(), IContainer.getABC()))