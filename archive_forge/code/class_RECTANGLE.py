import xcffib
import struct
import io
class RECTANGLE(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.x, self.y, self.width, self.height = unpacker.unpack('hhHH')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=hhHH', self.x, self.y, self.width, self.height))
        return buf.getvalue()
    fixed_size = 8

    @classmethod
    def synthetic(cls, x, y, width, height):
        self = cls.__new__(cls)
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        return self