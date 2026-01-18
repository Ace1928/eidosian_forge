import xcffib
import struct
import io
from . import xproto
class INT64(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.hi, self.lo = unpacker.unpack('iI')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=iI', self.hi, self.lo))
        return buf.getvalue()
    fixed_size = 8

    @classmethod
    def synthetic(cls, hi, lo):
        self = cls.__new__(cls)
        self.hi = hi
        self.lo = lo
        return self