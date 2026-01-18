import xcffib
import struct
import io
from . import xproto
from . import shm
class Rational(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.numerator, self.denominator = unpacker.unpack('ii')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=ii', self.numerator, self.denominator))
        return buf.getvalue()
    fixed_size = 8

    @classmethod
    def synthetic(cls, numerator, denominator):
        self = cls.__new__(cls)
        self.numerator = numerator
        self.denominator = denominator
        return self