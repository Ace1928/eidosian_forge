import xcffib
import struct
import io
from . import xproto
class SPANFIX(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.l, self.r, self.y = unpacker.unpack('iii')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=iii', self.l, self.r, self.y))
        return buf.getvalue()
    fixed_size = 12

    @classmethod
    def synthetic(cls, l, r, y):
        self = cls.__new__(cls)
        self.l = l
        self.r = r
        self.y = y
        return self