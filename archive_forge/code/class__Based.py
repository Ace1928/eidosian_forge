import unittest
class _Based:
    __bases__ = ()

    def __init__(self, name, bases=(), attrs=None):
        self.__name__ = name
        self.__bases__ = bases

    def __repr__(self):
        return self.__name__