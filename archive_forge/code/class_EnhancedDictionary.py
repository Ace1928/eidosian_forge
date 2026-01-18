import weakref
from inspect import isroutine
from copy import copy
from time import time
from kivy.eventmanager import MODE_DEFAULT_DISPATCH
from kivy.vector import Vector
class EnhancedDictionary(dict):

    def __getattr__(self, attr):
        try:
            return self.__getitem__(attr)
        except KeyError:
            return super(EnhancedDictionary, self).__getattr__(attr)

    def __setattr__(self, attr, value):
        self.__setitem__(attr, value)