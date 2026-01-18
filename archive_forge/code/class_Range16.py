import xcffib
import struct
import io
class Range16(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.first, self.last = unpacker.unpack('HH')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=HH', self.first, self.last))
        return buf.getvalue()
    fixed_size = 4

    @classmethod
    def synthetic(cls, first, last):
        self = cls.__new__(cls)
        self.first = first
        self.last = last
        return self