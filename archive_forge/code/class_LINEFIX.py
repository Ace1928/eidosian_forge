import xcffib
import struct
import io
from . import xproto
class LINEFIX(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.p1 = POINTFIX(unpacker)
        unpacker.pad(POINTFIX)
        self.p2 = POINTFIX(unpacker)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(self.p1.pack() if hasattr(self.p1, 'pack') else POINTFIX.synthetic(*self.p1).pack())
        buf.write(self.p2.pack() if hasattr(self.p2, 'pack') else POINTFIX.synthetic(*self.p2).pack())
        return buf.getvalue()

    @classmethod
    def synthetic(cls, p1, p2):
        self = cls.__new__(cls)
        self.p1 = p1
        self.p2 = p2
        return self