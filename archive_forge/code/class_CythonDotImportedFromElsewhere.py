from __future__ import absolute_import
import math, sys
class CythonDotImportedFromElsewhere(object):
    """
    cython.dataclasses just shadows the standard library modules of the same name
    """

    def __init__(self, module):
        self.__path__ = []
        self.__file__ = None
        self.__name__ = module
        self.__package__ = module

    def __getattr__(self, attr):
        from importlib import import_module
        import sys
        try:
            mod = import_module(self.__name__)
        except ImportError:
            raise AttributeError('%s: the standard library module %s is not available' % (attr, self.__name__))
        sys.modules['cython.%s' % self.__name__] = mod
        return getattr(mod, attr)