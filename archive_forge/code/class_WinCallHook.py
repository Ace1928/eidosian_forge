import ctypes
import functools
from winappdbg import compat
import sys
class WinCallHook(object):

    def __init__(self, dllname, funcname):
        self.__dllname = dllname
        self.__funcname = funcname
        self.__func = getattr(getattr(ctypes.windll, dllname), funcname)

    def __copy_attribute(self, attribute):
        try:
            value = getattr(self, attribute)
            setattr(self.__func, attribute, value)
        except AttributeError:
            try:
                delattr(self.__func, attribute)
            except AttributeError:
                pass

    def __call__(self, *argv):
        self.__copy_attribute('argtypes')
        self.__copy_attribute('restype')
        self.__copy_attribute('errcheck')
        print('-' * 10)
        print('%s ! %s %r' % (self.__dllname, self.__funcname, argv))
        retval = self.__func(*argv)
        print('== %r' % (retval,))
        return retval