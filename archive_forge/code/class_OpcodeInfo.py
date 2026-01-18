import codecs
import io
import pickle
import re
import sys
from struct import unpack as _unpack
from pickle import decode_long
class OpcodeInfo(object):
    __slots__ = ('name', 'code', 'arg', 'stack_before', 'stack_after', 'proto', 'doc')

    def __init__(self, name, code, arg, stack_before, stack_after, proto, doc):
        assert isinstance(name, str)
        self.name = name
        assert isinstance(code, str)
        assert len(code) == 1
        self.code = code
        assert arg is None or isinstance(arg, ArgumentDescriptor)
        self.arg = arg
        assert isinstance(stack_before, list)
        for x in stack_before:
            assert isinstance(x, StackObject)
        self.stack_before = stack_before
        assert isinstance(stack_after, list)
        for x in stack_after:
            assert isinstance(x, StackObject)
        self.stack_after = stack_after
        assert isinstance(proto, int) and 0 <= proto <= pickle.HIGHEST_PROTOCOL
        self.proto = proto
        assert isinstance(doc, str)
        self.doc = doc