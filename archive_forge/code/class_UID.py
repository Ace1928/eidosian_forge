import binascii
import codecs
import datetime
import enum
from io import BytesIO
import itertools
import os
import re
import struct
from xml.parsers.expat import ParserCreate
class UID:

    def __init__(self, data):
        if not isinstance(data, int):
            raise TypeError('data must be an int')
        if data >= 1 << 64:
            raise ValueError('UIDs cannot be >= 2**64')
        if data < 0:
            raise ValueError('UIDs must be positive')
        self.data = data

    def __index__(self):
        return self.data

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, repr(self.data))

    def __reduce__(self):
        return (self.__class__, (self.data,))

    def __eq__(self, other):
        if not isinstance(other, UID):
            return NotImplemented
        return self.data == other.data

    def __hash__(self):
        return hash(self.data)