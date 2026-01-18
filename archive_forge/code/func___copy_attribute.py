import ctypes
import functools
from winappdbg import compat
import sys
def __copy_attribute(self, attribute):
    try:
        value = getattr(self, attribute)
        setattr(self.__func, attribute, value)
    except AttributeError:
        try:
            delattr(self.__func, attribute)
        except AttributeError:
            pass