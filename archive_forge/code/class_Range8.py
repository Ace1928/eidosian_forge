import xcffib
import struct
import io
class Range8(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.first, self.last = unpacker.unpack('BB')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=BB', self.first, self.last))
        return buf.getvalue()
    fixed_size = 2

    @classmethod
    def synthetic(cls, first, last):
        self = cls.__new__(cls)
        self.first = first
        self.last = last
        return self