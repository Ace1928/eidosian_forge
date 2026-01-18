import itertools
from types import FunctionType
from zope.interface import Interface
from zope.interface import classImplements
from zope.interface.interface import InterfaceClass
from zope.interface.interface import _decorator_non_return
from zope.interface.interface import fromFunction
@staticmethod
def __is_reverse_protocol_name(name):
    return name.startswith('__r') and name.endswith('__')