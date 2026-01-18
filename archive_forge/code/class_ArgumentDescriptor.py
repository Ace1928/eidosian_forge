import codecs
import io
import pickle
import re
import sys
from struct import unpack as _unpack
from pickle import decode_long
class ArgumentDescriptor(object):
    __slots__ = ('name', 'n', 'reader', 'doc')

    def __init__(self, name, n, reader, doc):
        assert isinstance(name, str)
        self.name = name
        assert isinstance(n, int) and (n >= 0 or n in (UP_TO_NEWLINE, TAKEN_FROM_ARGUMENT1, TAKEN_FROM_ARGUMENT4, TAKEN_FROM_ARGUMENT4U, TAKEN_FROM_ARGUMENT8U))
        self.n = n
        self.reader = reader
        assert isinstance(doc, str)
        self.doc = doc