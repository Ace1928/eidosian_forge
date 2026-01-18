from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
class c_interop_string(c_char_p):

    def __init__(self, p=None):
        if p is None:
            p = ''
        if isinstance(p, str):
            p = p.encode('utf8')
        super(c_char_p, self).__init__(p)

    def __str__(self):
        return self.value

    @property
    def value(self):
        if super(c_char_p, self).value is None:
            return None
        return super(c_char_p, self).value.decode('utf8')

    @classmethod
    def from_param(cls, param):
        if isinstance(param, str):
            return cls(param)
        if isinstance(param, bytes):
            return cls(param)
        if param is None:
            return None
        raise TypeError("Cannot convert '{}' to '{}'".format(type(param).__name__, cls.__name__))

    @staticmethod
    def to_python_string(x, *args):
        return x.value