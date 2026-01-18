import sys
import unittest
import platform
import pygame
class Exporter2(Exporter):

    def get__array_interface__2(self):
        self.d = WRDict(Exporter.get__array_interface__(self))
        self.dict_ref = weakref.ref(self.d)
        return self.d
    __array_interface__ = property(get__array_interface__2)

    def free_dict(self):
        self.d = None

    def is_dict_alive(self):
        try:
            return self.dict_ref() is not None
        except AttributeError:
            raise NoDictError('__array_interface__ is unread')