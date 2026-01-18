from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
import collections
import warnings
class _RegistryContainer(object):
    """
    Base class for L{Registry} and L{RegistryKey}.
    """

    class __EmptyArgument:
        pass
    __emptyArgument = __EmptyArgument()

    def __init__(self):
        self.__default = None

    def has_key(self, name):
        return name in self

    def get(self, name, default=__emptyArgument):
        try:
            return self[name]
        except KeyError:
            if default is RegistryKey.__emptyArgument:
                return self.__default
            return default

    def setdefault(self, default):
        self.__default = default

    def __iter__(self):
        return compat.iterkeys(self)