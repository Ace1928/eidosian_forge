import xcffib
import struct
import io
from . import xproto
from . import shm
class Format(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.visual, self.depth = unpacker.unpack('IB3x')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=IB3x', self.visual, self.depth))
        return buf.getvalue()
    fixed_size = 8

    @classmethod
    def synthetic(cls, visual, depth):
        self = cls.__new__(cls)
        self.visual = visual
        self.depth = depth
        return self