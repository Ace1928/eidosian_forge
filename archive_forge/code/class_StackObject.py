import codecs
import io
import pickle
import re
import sys
from struct import unpack as _unpack
from pickle import decode_long
class StackObject(object):
    __slots__ = ('name', 'obtype', 'doc')

    def __init__(self, name, obtype, doc):
        assert isinstance(name, str)
        self.name = name
        assert isinstance(obtype, type) or isinstance(obtype, tuple)
        if isinstance(obtype, tuple):
            for contained in obtype:
                assert isinstance(contained, type)
        self.obtype = obtype
        assert isinstance(doc, str)
        self.doc = doc

    def __repr__(self):
        return self.name