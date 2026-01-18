import xcffib
import struct
import io
from . import xproto
class BufferAttributes(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.window, = unpacker.unpack('I')
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=I', self.window))
        return buf.getvalue()
    fixed_size = 4

    @classmethod
    def synthetic(cls, window):
        self = cls.__new__(cls)
        self.window = window
        return self