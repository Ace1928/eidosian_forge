import xcffib
import struct
import io
class CHAR2B(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.byte1, self.byte2 = unpacker.unpack('BB')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=BB', self.byte1, self.byte2))
        return buf.getvalue()
    fixed_size = 2

    @classmethod
    def synthetic(cls, byte1, byte2):
        self = cls.__new__(cls)
        self.byte1 = byte1
        self.byte2 = byte2
        return self