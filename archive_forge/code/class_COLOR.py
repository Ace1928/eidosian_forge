import xcffib
import struct
import io
from . import xproto
class COLOR(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.red, self.green, self.blue, self.alpha = unpacker.unpack('HHHH')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=HHHH', self.red, self.green, self.blue, self.alpha))
        return buf.getvalue()
    fixed_size = 8

    @classmethod
    def synthetic(cls, red, green, blue, alpha):
        self = cls.__new__(cls)
        self.red = red
        self.green = green
        self.blue = blue
        self.alpha = alpha
        return self