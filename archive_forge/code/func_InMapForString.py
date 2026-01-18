import array
import contextlib
import enum
import struct
def InMapForString(func):

    def wrapper(self, *args):
        if len(args) == 1:
            func(self, args[0])
        elif len(args) == 2:
            self.Key(args[0])
            func(self, args[1])
        else:
            raise ValueError('invalid number of arguments')
    return wrapper