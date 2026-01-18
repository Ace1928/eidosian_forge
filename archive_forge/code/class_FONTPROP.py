import xcffib
import struct
import io
class FONTPROP(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.name, self.value = unpacker.unpack('II')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=II', self.name, self.value))
        return buf.getvalue()
    fixed_size = 8

    @classmethod
    def synthetic(cls, name, value):
        self = cls.__new__(cls)
        self.name = name
        self.value = value
        return self