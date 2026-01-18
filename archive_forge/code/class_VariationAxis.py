import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
class VariationAxis(object):
    tag = None
    coords = tuple()

    def __init__(self, ftvaraxis):
        self.tag = unmake_tag(ftvaraxis.tag)
        self.name = ftvaraxis.name.decode('ascii')
        self.minimum = ftvaraxis.minimum / 65536.0
        self.default = ftvaraxis.default / 65536.0
        self.maximum = ftvaraxis.maximum / 65536.0
        self.strid = ftvaraxis.strid

    def __repr__(self):
        return "<VariationAxis '{}' ('{}') [{}, {}, {}]>".format(self.tag, self.name, self.minimum, self.default, self.maximum)