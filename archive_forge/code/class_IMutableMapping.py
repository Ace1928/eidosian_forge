import sys
from abc import ABCMeta
from collections import OrderedDict
from collections import UserDict
from collections import UserList
from collections import UserString
from collections import abc
from zope.interface.common import ABCInterface
from zope.interface.common import optional
class IMutableMapping(IMapping):
    abc = abc.MutableMapping
    extra_classes = (dict, UserDict)
    ignored_classes = (OrderedDict,)