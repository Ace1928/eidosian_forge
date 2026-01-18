import xcffib
import struct
import io
from . import xfixes
from . import xproto
class DeviceName(xcffib.Struct):
    xge = False

    def __init__(self, unpacker):
        if isinstance(unpacker, xcffib.Protobj):
            unpacker = xcffib.MemoryUnpacker(unpacker.pack())
        xcffib.Struct.__init__(self, unpacker)
        base = unpacker.offset
        self.len, = unpacker.unpack('B')
        self.string = xcffib.List(unpacker, 'c', self.len)
        self.bufsize = unpacker.offset - base

    def pack(self):
        buf = io.BytesIO()
        buf.write(struct.pack('=B', self.len))
        buf.write(xcffib.pack_list(self.string, 'c'))
        return buf.getvalue()

    @classmethod
    def synthetic(cls, len, string):
        self = cls.__new__(cls)
        self.len = len
        self.string = string
        return self